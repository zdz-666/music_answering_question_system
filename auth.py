"""账号体系：用「用户名 + 密码」换回稳定的 user_id。

改造前 user_id 是前端随机生成、只存在 localStorage 里的——换浏览器或清缓存就变成
另一个人，会话与长期记忆全部对不上。这里补一层最小可用的账号：

- 账号数据存 Redis（与会话同源）。用 cache.require_client()，拿不到连接就报错、
  不做降级：账号读不出来却装作「登录失败」，会让人以为自己记错了密码；
- 密码只存 pbkdf2-hmac-sha256 的加盐哈希，用标准库实现，不引入新依赖；
- user_id 由账号派生（`user_<规范化用户名>`），于是同一账号在任何浏览器登录，
  看到的都是同一批会话与同一份跨会话记忆。

定位说明：这是**身份识别**，不是**访问控制**。登录只负责证明「你是这个账号」并把它的
user_id 交回前端；后续请求依旧按前端带来的 user_id 归属数据（与改造前一致，没有引入新的
越权面）。真正的鉴权要在每个端点上校验 token，不在本次范围内。

键设计（前缀复用 cache.key()）：

    {prefix}:user:{digest(username)}   HASH   user_id / username / salt / password_hash / created_at

账号不设 TTL：会话与记忆都挂在这个身份上，账号过期会让它们变成无主数据。
"""

import hashlib
import hmac
import re
import secrets
from datetime import datetime

import cache
from config import AUTH_PBKDF2_ITERATIONS
from observability import log_step, logger

_PURPOSE = "账号存储"

# 用户名规则：3-32 位小写字母 / 数字 / 下划线 / 点 / 连字符。
# 收紧字符集不是为了安全（记忆目录另有 safe_dir 兜底），而是让 user_id 与目录名可读、
# 可排查，避免用户名里出现空格、斜杠之类的意外字符。
_USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")

_SALT_BYTES = 16
_MIN_PASSWORD_LEN = 6
_MAX_PASSWORD_LEN = 128

# 注册要「检查是否已存在 + 写入全部字段」两步，拆开会在并发注册同名账号时
# 出现「user_id 写进去了、密码哈希没写」的半条记录；放进一段 Lua 里原子完成。
_REGISTER_LUA = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
redis.call('HSET', KEYS[1],
    'user_id', ARGV[1],
    'username', ARGV[2],
    'salt', ARGV[3],
    'password_hash', ARGV[4],
    'created_at', ARGV[5])
return 1
"""

_register_script = None


class AuthError(ValueError):
    """注册 / 登录失败的原因，由入口层按 status 映射成 400 / 409 / 401。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _decode(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def normalize_username(raw: str) -> str:
    """用户名规范化：去首尾空白 + 转小写（`Alice` 与 `alice` 视为同一账号）。"""
    return (raw or "").strip().lower()


def derive_user_id(username: str) -> str:
    """账号 → user_id。带 `user_` 前缀，沿用改造前前端生成 id 的形状。"""
    return f"user_{username}"


def _user_key(username: str) -> str:
    """账号记录键。用 digest 压掉用户名里的特殊字符，避免与键的 `:` 结构混淆。"""
    return cache.key("user", cache.digest(username))


def _hash_password(password: str, salt: str) -> str:
    """pbkdf2-hmac-sha256 → 十六进制；salt 也是十六进制字符串。"""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        AUTH_PBKDF2_ITERATIONS,
    ).hex()


def _validate(username: str, password: str) -> None:
    if not _USERNAME_RE.match(username):
        raise AuthError("用户名需为 3-32 位字母、数字、下划线、点或连字符", 400)
    if len(password) < _MIN_PASSWORD_LEN:
        raise AuthError(f"密码至少 {_MIN_PASSWORD_LEN} 位", 400)
    if len(password) > _MAX_PASSWORD_LEN:
        raise AuthError("密码过长", 400)


def register(username: str, password: str) -> dict:
    """注册新账号，返回 {"user_id", "username"}；账号已存在时抛 AuthError(409)。"""
    global _register_script

    username = normalize_username(username)
    password = password or ""
    _validate(username, password)

    client = cache.require_client(_PURPOSE)
    user_id = derive_user_id(username)
    salt = secrets.token_hex(_SALT_BYTES)

    if _register_script is None:
        _register_script = client.register_script(_REGISTER_LUA)

    with log_step("auth.register", username=username) as span:
        created = _register_script(
            keys=[_user_key(username)],
            args=[
                user_id,
                username,
                salt,
                _hash_password(password, salt),
                datetime.now().isoformat(),
            ],
        )
        if not int(created or 0):
            # 日志里只记账号名，不记密码
            logger.info("auth.register.duplicate", extra={"fields": {"username": username}})
            raise AuthError("账号已存在", 409)
        span.output = {"user_id": user_id}

    return {"user_id": user_id, "username": username}


def authenticate(username: str, password: str) -> dict:
    """校验账号密码，成功返回 {"user_id", "username"}；失败抛 AuthError(401)。

    「账号不存在」与「密码不对」返回同一句提示，避免把账号是否存在探出来。
    """
    username = normalize_username(username)
    client = cache.require_client(_PURPOSE)

    with log_step("auth.login", username=username) as span:
        record = client.hgetall(_user_key(username))
        if not record:
            span.output = {"ok": False, "reason": "unknown_user"}
            raise AuthError("账号或密码不正确", 401)

        record = {_decode(key): _decode(value) for key, value in record.items()}
        expected = record.get("password_hash", "")
        actual = _hash_password(password or "", record.get("salt", ""))
        if not hmac.compare_digest(expected, actual):
            span.output = {"ok": False, "reason": "bad_password"}
            raise AuthError("账号或密码不正确", 401)

        user_id = record.get("user_id") or derive_user_id(username)
        span.output = {"ok": True, "user_id": user_id}

    return {"user_id": user_id, "username": record.get("username") or username}