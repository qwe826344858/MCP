from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import ToolsAPI.weather as w

# 初始化 FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"

####################################################################################################
#  tool execution
#  tool Handlers

@mcp.tool()
async def get_alerts(state: str) -> str:
    """获取美国州的天气警报。

    Args:
        state: 两个字母的美国州代码（例如 CA、NY）
    """
    print("get_alerts start")
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await w.make_nws_request(url)

    if not data or "features" not in data:
        return "无法获取警报或未找到警报。"

    if not data["features"]:
        return "该州没有活跃的警报。"

    alerts = [w.format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """获取某个位置的天气预报。

    Args:
        latitude: 位置的纬度
        longitude: 位置的经度
    """
    print("get_forecast start")
    # 首先获取预报网格 endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await w.make_nws_request(points_url)

    if not points_data:
        return "无法获取此位置的预报数据。"

    # 从 points response 中获取预报 URL
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await w.make_nws_request(forecast_url)

    if not forecast_data:
        return "无法获取详细预报。"

    # 将 periods 格式化为可读的预报
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # 仅显示接下来的 5 个 periods
        forecast = f"""
{period['name']}:
温度: {period['temperature']}°{period['temperatureUnit']}
风: {period['windSpeed']} {period['windDirection']}
预报: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


if __name__ == "__main__":
    # 初始化并运行 server
    mcp.run(transport='stdio')
