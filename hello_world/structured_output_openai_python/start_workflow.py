import asyncio
import sys
import uuid
import os
from datetime import datetime
from typing import Any
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.clean_data_workflow import CleanDataWorkflow


async def main():
    # Connect to Temporal server with matching data converter
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    # Get data from command line or use realistic messy business data
    if len(sys.argv) > 1:
        data = " ".join(sys.argv[1:])
    else:
        # Very messy, realistic business data with missing/invalid information
        data = """
        TECH SOLUTIONS INC.
        Email: contact@techsolutions.net (primary) / sales@techsolutions.net 
        Phone: (555) 123-4567 ext. 101 // Call us! Also try our emergency line: not-a-phone-number
        Address: 1234 Innovation Dr, Suite 200, San Francisco, CA 94107-1234
        Website: www.techsolutions.net | https://techsolutions.net | broken-link.com/404
        Industry: Software Development & IT Services
        
        ---
        
        Green Valley Eats 
        contact email: info@greenvalleyeats.com
        Tel: MISSING - ask owner
        Location: 789 Example Street, Austin TX 78701  
        Web: TODO - look up later
        Type of Business: Restaurant/Food Service
        
        ---
        
        EXAMPLE & ASSOCIATES LAW FIRM
        E-mail: NONE PROVIDED
        Phone Number: +1-555-999-LAWW
        Office Address: 456 Legal Plaza, Floor 15, Boston, MA 02108
        Website URL: https://examplelaw.org/
        Practice Area: Legal Services & More Legal Stuff
        
        ---
        
        Bloom Flower Shop
        Email Address: orders@bloomflowers.co
        Phone: 555 234 5678 ext 349i
        Street Address: unknown - somewhere on Garden Ave, Portland, OR 
        Online: bloomflowers.co (down for maintenance)
        Business Type: Retail - Flowers & Gifts
        
        ---
        
        DataCorp Analytics LLC  
        Primary Email: hello@datacorp.io
        Contact Phone: (555) 345-6789
        Business Address: 567 Data Drive, Suite 300, Seattle, WA 98101
        Company Website: https://datacorp.io
        Industry Sector: Data Analytics & Business Intelligence
        
        ---
        
        Bob's Auto Repair
        email: bob@email  
        phone: 5553924302
        address: [REDACTED FOR PRIVACY]
        website: N/A
        about: fix cars and trucks
        
        ---
        
        THE BEST PIZZA CO!!!
        Contact Info: Missing! You can look it up in the yellow pages.
        Location: Somewhere downtown  
        Hours: whenever we feel like it
        Food: AMAZING PIZZA 🍕
        """

    print("Starting Clean Data workflow...")
    print("=" * 80)

    # Generate unique workflow ID (or use environment variable if set)
    workflow_id = os.environ.get("WORKFLOW_ID")
    if not workflow_id:
        workflow_id = f"clean-data-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    print(f"Workflow ID: {workflow_id}")

    def format(x: Any) -> str:
        if business is None:
            return "MISSING"
        return str(x)

    try:
        # Execute the clean data workflow
        result = await client.execute_workflow(
            CleanDataWorkflow.run,
            data,
            id=workflow_id,
            task_queue="clean-data-task-queue",
        )

        print("CLEAN DATA COMPLETED!")
        print("=" * 80)
        print("Cleaned business data:")
        for i, business in enumerate(result.businesses, 1):
            print(f"\n{i}. {format(business.name)}")
            print(f"   Email: {format(business.email)}")
            print(f"   Phone: {format(business.phone)}")
            print(f"   Address: {format(business.address)}")
            print(f"   Website: {format(business.website)}")
            print(f"   Industry: {format(business.industry)}")

    except Exception as e:
        print(f"Clean Data failed: {e}")
        print("Please check that:")
        print("1. Temporal server is running (temporal server start-dev)")
        print("2. Worker is running (uv run python -m worker)")
        print("3. OpenAI API key is configured")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
