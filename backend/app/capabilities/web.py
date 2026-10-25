"""Bounded public-web access. Each connection is pinned to a checked DNS address."""
import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup


def public_url(value: str) -> httpx.URL:
    if any(ord(char) < 32 for char in value) or "\\" in value:
        raise ValueError("Use a public HTTP or HTTPS URL.")
    url = httpx.URL(value)
    if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
            or url.port not in {None, 80, 443}):
        raise ValueError("Only public HTTP/HTTPS pages on ports 80 or 443 are supported.")
    return url


async def public_address(host: str, port: int) -> str:
    addresses = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ips = [ipaddress.ip_address(item[4][0]) for item in addresses]
    if not ips or any(not ip.is_global or (getattr(ip, "ipv4_mapped", None)
                                          and not ip.ipv4_mapped.is_global) for ip in ips):
        raise ValueError("Local, private and reserved network addresses are not accessible.")
    return str(ips[0])


async def read_webpage(url: str, start_char: int = 0) -> dict:
    current = public_url(url)
    async with asyncio.timeout(25):
        async with httpx.AsyncClient(timeout=15, follow_redirects=False, trust_env=False) as client:
            for _ in range(6):
                ip = await public_address(current.host, current.port or (443 if current.scheme == "https" else 80))
                # Preserve HTTP Host and TLS certificate/SNI hostname while connecting
                # directly to the validated IP, preventing a second DNS lookup/rebinding.
                pinned = current.copy_with(host=ip)
                async with client.stream("GET", pinned,
                                         headers={"Host": current.netloc.decode("ascii"),
                                                  "User-Agent": "AgenticFriend/1.0 (web reader)",
                                                  "Accept-Encoding": "identity",
                                                  "Accept": "text/html,text/plain"},
                                         extensions={"sni_hostname": current.host}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("The page redirected without a destination.")
                        current = public_url(urljoin(str(current), location))
                        continue
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ValueError("The server ignored the uncompressed-page request; this page cannot be read safely within the size limit.")
                    content_type = response.headers.get("content-type", "").lower()
                    if not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml+xml")):
                        raise ValueError("This reader supports HTML and plain text, not PDFs, images or downloads.")
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        body.extend(chunk)
                        if len(body) > 1_000_000:
                            raise ValueError("Page exceeds the 1 MB reading limit.")
                    text = bytes(body).decode(response.encoding or "utf-8", errors="replace")
                    return extract_page(text, str(current), start_char, "html" in content_type)
    raise ValueError("Too many webpage redirects.")


def extract_page(body: str, url: str, start_char: int, html: bool = True) -> dict:
    title, links = url, []
    if html:
        soup = BeautifulSoup(body, "html.parser")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)[:250]
        for node in soup(["script", "style", "noscript", "template", "nav", "footer", "header", "svg"]):
            node.decompose()
        main = soup.find("main") or soup.find("article") or soup
        for anchor in main.find_all("a", href=True):
            candidate = urljoin(url, anchor["href"])
            if urlsplit(candidate).scheme in {"http", "https"}:
                links.append({"title": anchor.get_text(" ", strip=True)[:150], "url": candidate[:2048]})
            if len(links) == 15:
                break
        text = main.get_text("\n", strip=True)
    else:
        text = body
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())[:200000]
    end = start_char + 6000
    return {"url": url, "title": title, "text": text[start_char:end], "start_char": start_char,
            "next_start_char": end if end < len(text) else None, "links": links,
            "untrusted_content": True}


def web_search(query: str) -> dict:
    # No API key required. Provider throttling is reported as a tool failure;
    # search snippets are never presented as pages we actually read.
    from ddgs import DDGS

    results = DDGS(timeout=12).text(query, max_results=5, backend="duckduckgo,bing")
    items = []
    for item in results:
        try:
            url = str(public_url(item.get("href", "")))
        except (ValueError, httpx.InvalidURL):
            continue
        items.append({"title": item.get("title", "")[:250], "url": url,
                      "snippet": item.get("body", "")[:600]})
    if not items:
        raise ValueError("Search returned no results. Try a more specific query or read a known public URL.")
    return {"query": query, "results": items, "untrusted_content": True}
