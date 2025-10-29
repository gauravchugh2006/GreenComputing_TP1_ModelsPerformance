import asyncio
from pyppeteer import launch

async def save_streamlit_as_html():
    browser = await launch()
    page = await browser.newPage()
    await page.goto('http://localhost:8501', {'waitUntil': 'networkidle2'})
    await page.pdf({'path': 'streamlit_dashboard.pdf'})
    await page.screenshot({'path': 'streamlit_dashboard.png'})
    await page.content()
    html = await page.content()
    with open('streamlit_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html)
    await browser.close()

asyncio.get_event_loop().run_until_complete(save_streamlit_as_html())