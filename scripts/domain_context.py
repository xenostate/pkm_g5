#!/usr/bin/env python3
"""
Domain-scoped storage helpers for multi-subject PKM workspaces.
"""

import json
import re
import shutil
from contextvars import ContextVar
from pathlib import Path

_current_domain: ContextVar[str] = ContextVar("pkm_current_domain", default="general")

BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOMAINS_DIR = BASE_DATA_DIR / "domains"
LEGACY_KB_PATH = BASE_DATA_DIR / "knowledge_base.json"
LEGACY_CHROMA_DIR = BASE_DATA_DIR / "chroma_db"


def normalize_domain(value: str | None) -> str:
    raw = (value or "general").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or "general"


def set_current_domain(value: str | None):
    return _current_domain.set(normalize_domain(value))


def reset_current_domain(token):
    _current_domain.reset(token)


def get_current_domain() -> str:
    return _current_domain.get()


def domain_dir(domain: str | None = None) -> Path:
    slug = normalize_domain(domain) if domain is not None else get_current_domain()
    path = DOMAINS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def domain_data_dir(domain: str | None = None) -> Path:
    return domain_dir(domain)


def domain_meta_path(domain: str | None = None) -> Path:
    return domain_dir(domain) / "meta.json"


def ensure_domain(domain_name: str) -> dict:
    slug = normalize_domain(domain_name)
    meta_path = domain_meta_path(slug)
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps({"id": slug, "name": domain_name.strip() or slug}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("name"):
            meta["name"] = domain_name.strip() or slug
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_domain_meta(slug)


def get_domain_meta(domain: str) -> dict:
    slug = normalize_domain(domain)
    meta_path = domain_meta_path(slug)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"id": slug, "name": meta.get("name", slug)}
    return {"id": slug, "name": slug}


def list_domains() -> list[dict]:
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    domains = []
    for path in DOMAINS_DIR.iterdir():
        if not path.is_dir():
            continue
        domains.append(get_domain_meta(path.name))

    if not domains:
        domains.append(ensure_domain("General"))

    domains.sort(key=lambda item: item["name"].lower())
    return domains


def migrate_legacy_data_to_general() -> bool:
    """Copy old single-workspace data into the General domain on first migration."""
    general_dir = domain_dir("general")
    general_kb_path = general_dir / "knowledge_base.json"
    general_chroma_dir = general_dir / "chroma_db"
    migrated = False

    if LEGACY_KB_PATH.exists() and not general_kb_path.exists():
        shutil.copy2(LEGACY_KB_PATH, general_kb_path)
        migrated = True

    if LEGACY_CHROMA_DIR.exists() and not general_chroma_dir.exists():
        shutil.copytree(LEGACY_CHROMA_DIR, general_chroma_dir)
        migrated = True

    if migrated:
        ensure_domain("General")

    return migrated
