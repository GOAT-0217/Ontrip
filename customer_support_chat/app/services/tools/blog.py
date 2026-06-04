# customer_support_chat/app/services/tools/blog.py

import httpx
import time
from datetime import datetime
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings

settings = get_settings()

"""
小红书（Xiaohongshu / RedNote）攻略搜索工具
============================================
通过 Apify 平台调用小红书搜索 Actor（zen-studio/rednote-search-scraper），
实现类似在 APP 中搜索"XX城市旅游攻略""XX景点推荐"等关键词的功能。

Actor: zen-studio~rednote-search-scraper
  免费额度: 10 次试用运行
  定价: $0.05/次启动 + $0.007/条结果
  速度: ~4 秒返回 5 条结果

工作流程：
  1. 向 Apify 提交搜索任务（POST /runs）
  2. 轮询任务状态直至完成（GET /runs/{runId}）
  3. 从 run data 中获取 defaultDatasetId
  4. 获取结构化笔记数据（GET /datasets/{datasetId}/items）
  5. 格式化为 Markdown 返回
"""


def _start_apify_run(client: httpx.Client, keyword: str, limit: int) -> str:
    """向 Apify 提交搜索任务，返回 run_id。"""
    actor_id = settings.XHS_ACTOR_ID
    url = f"https://api.apify.com/v2/acts/{actor_id}/runs"

    payload = {
        "keywords": [keyword],                  # 关键词数组
        "maxResults": limit,                    # 最大结果数
        "sortType": "popularity_descending",    # 按点赞数排序，取最热门的
        "noteType": "all",                      # 所有类型笔记
    }

    headers = {
        "Authorization": f"Bearer {settings.APIFY_API_TOKEN}",
        "Content-Type": "application/json",
    }

    resp = client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    run_id = data.get("data", {}).get("id", "")
    if not run_id:
        raise RuntimeError(f"Apify 未返回 run_id，响应: {data}")
    return run_id


def _poll_run(client: httpx.Client, run_id: str, timeout: int = 60) -> dict:
    """轮询 Apify 任务状态直到完成，返回 run 详情（含 defaultDatasetId）。"""
    actor_id = settings.XHS_ACTOR_ID
    url = f"https://api.apify.com/v2/acts/{actor_id}/runs/{run_id}"

    headers = {
        "Authorization": f"Bearer {settings.APIFY_API_TOKEN}",
    }

    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        status = data.get("status", "")

        if status == "SUCCEEDED":
            return data
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Apify 任务 {run_id} 状态为 {status}")

        time.sleep(2)

    raise TimeoutError(f"Apify 任务 {run_id} 在 {timeout}s 内未完成")


def _fetch_dataset(client: httpx.Client, dataset_id: str, limit: int) -> list[dict]:
    """从 Apify Dataset 获取搜索结果。"""
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    headers = {
        "Authorization": f"Bearer {settings.APIFY_API_TOKEN}",
    }

    resp = client.get(url, headers=headers)
    resp.raise_for_status()
    items = resp.json()

    if isinstance(items, list):
        return items[:limit]
    return []


def _format_note(note: dict, index: int) -> str:
    """将一条小红书笔记格式化为 Markdown 卡片。"""
    title = note.get("title") or "无标题"
    desc = note.get("desc") or ""
    url = note.get("url") or ""

    # 作者信息（嵌套对象）
    author = note.get("author", {})
    nickname = author.get("nickname") if isinstance(author, dict) else ""

    # 互动数据（嵌套对象）
    engagement = note.get("engagement", {})
    likes = engagement.get("liked_count", 0) if isinstance(engagement, dict) else 0
    collects = engagement.get("collected_count", 0) if isinstance(engagement, dict) else 0

    # 时间戳转换
    ts = note.get("timestamp")
    time_str = ""
    if ts:
        try:
            dt = datetime.fromtimestamp(ts)
            time_str = dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Markdown 卡片：完整正文 + 链接
    lines = [f"**{index}. {title}**"]
    meta = []
    if nickname:
        meta.append(f"👤 {nickname}")
    if likes:
        meta.append(f"❤️ {likes}")
    if collects:
        meta.append(f"⭐ {collects}")
    if time_str:
        meta.append(f"🕐 {time_str}")
    if meta:
        lines.append(" · ".join(meta))
    if desc:
        lines.append("")  # 空行隔开
        lines.append(desc)
    if url:
        lines.append("")
        lines.append(f"🔗 [打开小红书查看原文]({url})")

    return "\n".join(lines)


@tool
def search_xiaohongshu(keyword: str) -> str:
    """搜索小红书（Xiaohongshu / RedNote）上的旅行攻略，按点赞数排序，返回前 3 条最热门的笔记。

    当用户询问以下类型的问题时调用此工具：
      - "XX有什么好玩的" / "XX旅游攻略" / "XX景点推荐" / "XX旅行计划"
      - "XX有什么好吃的" / "XX美食推荐"
      - "XX酒店推荐" / "XX民宿测评"
      - "XX拍照打卡" / "XX小众景点"

    Note: 本工具固定返回点赞最多的 3 条笔记。

    Args:
        keyword: 搜索关键词，例如 "北京三日游攻略"、"三亚酒店推荐"、"成都美食必吃"

    Returns:
        Markdown 格式的小红书笔记，包含标题、正文（搜索接口返回的完整摘要）、点赞收藏数、链接。
    """
    if not settings.APIFY_API_TOKEN:
        return (
            "❌ 小红书搜索功能尚未配置 API Token。\n"
            "请按以下步骤配置：\n"
            "1. 访问 https://apify.com 注册账号\n"
            "2. 进入 Settings → Integrations 复制 API Token\n"
            "3. 将 Token 填入环境变量 APIFY_API_TOKEN\n"
            "4. 在 Apify Store 搜索 'RedNote Search Scraper' 并安装（首次10次试用免费）"
        )

    limit = 3  # 固定 3 条，LLM 不能绕过

    with httpx.Client(timeout=90) as client:
        try:
            # 1) 启动搜索任务
            run_id = _start_apify_run(client, keyword, limit)

            # 2) 等待任务完成，获取 defaultDatasetId
            run_data = _poll_run(client, run_id, timeout=60)
            dataset_id = run_data.get("defaultDatasetId", "")
            if not dataset_id:
                raise RuntimeError("Apify 未返回 defaultDatasetId")

            # 3) 获取结果
            notes = _fetch_dataset(client, dataset_id, limit)

            if not notes:
                return (
                    f"🔍 在小红书上搜索「{keyword}」未找到相关笔记。\n"
                    "建议更换关键词尝试，例如：\n"
                    f"  - 「{keyword}推荐」\n"
                    f"  - 「{keyword}攻略」\n"
                    f"  - 「{keyword}必去」"
                )

            # 4) 格式化输出
            results = [_format_note(note, i + 1) for i, note in enumerate(notes)]
            header = f"===== 📕 小红书搜索「{keyword}」共 {len(notes)} 条 =====\n"
            footer = "\n===== 搜索结果来自小红书，仅供参考 ====="
            return header + "\n\n".join(results) + footer

        except TimeoutError as e:
            return f"⏰ 小红书搜索超时，请稍后重试: {e}"
        except httpx.HTTPStatusError as e:
            return f"❌ Apify API 请求失败（HTTP {e.response.status_code}）: {e}"
        except RuntimeError as e:
            return f"❌ 小红书搜索任务异常: {e}"
        except Exception as e:
            return f"❌ 小红书搜索发生未知错误: {e}"
