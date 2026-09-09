from unittest.mock import MagicMock, patch

import pytest
from temporalio.testing import ActivityEnvironment

from activities.tools import get_recent_aws_announcements

SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>AWS What's New</title>
    <item>
      <title>Amazon S3 announces a new feature</title>
      <link>https://aws.amazon.com/about-aws/whats-new/2026/01/s3-feature/</link>
      <pubDate>Mon, 05 Jan 2026 17:00:00 +0000</pubDate>
    </item>
    <item>
      <title>AWS Lambda adds support for something</title>
      <link>https://aws.amazon.com/about-aws/whats-new/2026/01/lambda-feature/</link>
      <pubDate>Tue, 06 Jan 2026 17:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Amazon EC2 introduces a new instance type</title>
      <link>https://aws.amazon.com/about-aws/whats-new/2026/01/ec2-instance/</link>
      <pubDate>Wed, 07 Jan 2026 17:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


def _mock_response():
    response = MagicMock()
    response.content = SAMPLE_FEED
    response.raise_for_status = MagicMock()
    return response


@pytest.mark.asyncio
async def test_parses_announcements():
    with patch("requests.get", return_value=_mock_response()):
        result = await ActivityEnvironment().run(get_recent_aws_announcements)

    # Default limit is 5; the sample feed only has 3 items.
    assert len(result) == 3
    assert result[0] == {
        "title": "Amazon S3 announces a new feature",
        "link": "https://aws.amazon.com/about-aws/whats-new/2026/01/s3-feature/",
        "published": "Mon, 05 Jan 2026 17:00:00 +0000",
    }


@pytest.mark.asyncio
async def test_respects_limit():
    with patch("requests.get", return_value=_mock_response()):
        result = await ActivityEnvironment().run(get_recent_aws_announcements, 2)

    assert len(result) == 2
    assert [item["title"] for item in result] == [
        "Amazon S3 announces a new feature",
        "AWS Lambda adds support for something",
    ]


@pytest.mark.asyncio
async def test_raises_on_http_error():
    response = _mock_response()
    response.raise_for_status.side_effect = RuntimeError("503 Service Unavailable")

    with patch("requests.get", return_value=response):
        with pytest.raises(RuntimeError):
            await ActivityEnvironment().run(get_recent_aws_announcements)
