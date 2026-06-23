import asyncio
import xml.etree.ElementTree as ET

from temporalio import activity

# AWS publishes a live "What's New" RSS feed of recent service launches and updates.
WHATS_NEW_FEED = "https://aws.amazon.com/about-aws/whats-new/recent/feed/"


@activity.defn
async def get_recent_aws_announcements(limit: int = 5) -> list[dict]:
    """Fetch the most recent AWS 'What's New' announcements from the live RSS feed.

    Use this when the user asks what is new or recently launched in AWS. Returns a
    list of {title, link, published} for the latest service launches and updates.
    """
    # Import requests lazily here (not at module top level) so the workflow can
    # import this activity without pulling a non-deterministic module into the
    # workflow sandbox. requests is blocking, so run it off the event loop;
    # Temporal handles retries, so no client-side retry configuration is needed.
    import requests

    response = await asyncio.to_thread(requests.get, WHATS_NEW_FEED, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    return [
        {
            "title": item.findtext("title", ""),
            "link": item.findtext("link", ""),
            "published": item.findtext("pubDate", ""),
        }
        for item in root.findall(".//item")[:limit]
    ]
