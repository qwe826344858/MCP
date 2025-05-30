from typing import Any
import httpx

USER_AGENT = "weather-app/1.0"

####################################################################################################
#  Helper functions
# API Handlers
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """向 NWS API 发送请求，并进行适当的错误处理。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    print("make_nws_request start")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """将警报 feature 格式化为可读的字符串。"""
    print("format_alert start")
    props = feature["properties"]
    return f"""
事件: {props.get('event', 'Unknown')}
区域: {props.get('areaDesc', 'Unknown')}
严重性: {props.get('severity', 'Unknown')}
描述: {props.get('description', 'No description available')}
指示: {props.get('instruction', 'No specific instructions provided')}
"""
