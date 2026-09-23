"""一键把当前代码同步到魔搭创空间。

创空间是独立 git 仓库（不从 GitHub 自动同步），所以改完代码要手动同步一次：

    python scripts/sync_studio.py            # 只同步代码
    python scripts/sync_studio.py --deploy   # 同步并重新部署

令牌取自环境变量 MODELSCOPE_API_TOKEN（或项目根目录的 .env），
在 https://modelscope.cn/my/myaccesstoken 生成，需要读写权限。
"""
from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "EstrellaSky/ev-advisor"
IGNORE_PATTERNS = (
    ".git",
    ".git/*",
    ".env",
    ".ms_upload_cache",
    ".venv/*",
    "venv/*",
    "__pycache__",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    ".pytest_cache",
    ".pytest_cache/*",
    "**/.pytest_cache/*",
    "data/processed/embeddings",
    "data/processed/embeddings/*",
    "*.log",
    "logs",
)


def read_token() -> str | None:
    """环境变量优先，其次读项目根目录的 .env。"""
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        return token.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MODELSCOPE_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="同步代码到魔搭创空间")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="创空间 ID，形如 owner/name")
    parser.add_argument("--deploy", action="store_true", help="同步后立即重新部署")
    parser.add_argument("--message", default="chore: 同步最新代码到创空间", help="提交信息")
    args = parser.parse_args()

    token = read_token()
    if not token:
        print("缺少 MODELSCOPE_API_TOKEN：请在 .env 里配置，或先设置该环境变量。")
        return 1

    from modelscope_hub import HubApi

    api = HubApi(token=token)
    tracker = Path(tempfile.gettempdir()) / f"ms-sync-{uuid.uuid4().hex[:8]}.json"
    print(f"同步 {ROOT} -> {args.repo} ...")
    result = api.upload_folder(
        repo_id=args.repo,
        repo_type="studio",
        folder_path=str(ROOT),
        ignore_patterns=list(IGNORE_PATTERNS),
        commit_message=args.message,
        tracker_path=str(tracker),
        disable_tqdm=True,
    )
    print("已提交:", (result or {}).get("commit_id") if isinstance(result, dict) else result)

    if args.deploy:
        # 带上 payload 强制重建：不带时平台可能直接复用上一次的镜像，
        # 表现是「几十秒就构建成功、但线上还是旧代码」（实测踩过）。
        api.deploy_repo(
            args.repo,
            "studio",
            payload={"no_cache": True, "rebuild": True, "revision": "master"},
        )
        print("已触发重新部署，构建通常需要 3-8 分钟（改了依赖会更久）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
