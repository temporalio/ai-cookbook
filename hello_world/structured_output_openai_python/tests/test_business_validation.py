from workflows.clean_data_workflow import Business
import pytest
from pydantic import ValidationError


class TestBusiness:
    def test_valid_business(self):
        business = Business(
            name="Acme Corporation",
            email="info@acmecorp.com",
            phone="+12025550173",
            address="123 Business Park Dr, Suite 100, New York, NY 10001",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business.name == "Acme Corporation"
        assert business.email == "info@acmecorp.com"
        assert business.phone == "+12025550173"
        assert business.address == "123 Business Park Dr, Suite 100, New York, NY 10001"
        assert business.website == "https://www.acmecorp.com"
        assert business.industry == "Technology"

    def test_business_name_whitespace_stripping(self):
        business = Business(
            name="  Acme Corp  ",
            email="info@acmecorp.com",
            phone="+12025550173",
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business.name == "Acme Corp"

    def test_phone_number_strict_e164_validation(self):
        # E.164 regex validation only accepts properly formatted E.164 numbers
        valid_e164_numbers = [
            "+12025550173",
            "+442079460958",
            "+33123456789",
            "+81312345678",
            "+61234567890",
            "+49301234567",
        ]

        for e164_phone in valid_e164_numbers:
            business = Business(
                name="Acme Corp",
                email="info@acmecorp.com",
                phone=e164_phone,
                address="123 Business Park Dr",
                website="https://www.acmecorp.com",
                industry="Technology",
            )
            assert business.phone == e164_phone

    def test_email_case_normalized(self):
        # Email case is normalized to lowercase by pydantic's EmailStr
        business = Business(
            name="Acme Corp",
            email="INFO@ACMECORP.COM",
            phone="+12025550173",
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business.email == "info@acmecorp.com"

    def test_optional_fields(self):
        # All fields are optional now - these should all work
        business_empty: Business = Business()  # type: ignore
        assert business_empty.name is None
        assert business_empty.email is None
        assert business_empty.phone is None

        business_partial: Business = Business(name="Acme Corp")  # type: ignore
        assert business_partial.name == "Acme Corp"
        assert business_partial.email is None

        business_two_fields: Business = Business(
            name="Acme Corp", email="info@acmecorp.com"
        )  # type: ignore
        assert business_two_fields.name == "Acme Corp"
        assert business_two_fields.email == "info@acmecorp.com"
        assert business_two_fields.phone is None

    def test_invalid_email_formats_raise_errors(self):
        # Invalid emails should raise ValidationError instead of being converted to None
        invalid_emails = [
            "invalid-email",
            "@acmecorp.com",
            "info@",
            "info name@acmecorp.com",
            "info@.com",
        ]

        for invalid_email in invalid_emails:
            with pytest.raises(ValidationError):
                Business(
                    name="Acme Corp",
                    email=invalid_email,
                    phone="+12025550173",
                    address="123 Business Park Dr",
                    website="https://www.acmecorp.com",
                    industry="Technology",
                )

    def test_invalid_phone_numbers_raise_errors(self):
        # Invalid phones should raise ValidationError instead of being converted to None
        invalid_phones = [
            ("123456789", "Missing plus sign and too short"),
            ("+12345678901234567", "Too long - 17 digits total"),
            ("not-a-phone", "Invalid format"),
            ("123", "Way too short"),
            ("abc-def-ghij", "Letters only"),
            ("+0123456789", "Starts with 0 after plus"),
            ("+1 202 555 0173", "Contains spaces"),
            ("+1-202-555-0173", "Contains hyphens"),
            ("202-555-0173", "Missing plus and country code"),
            ("(202) 555-0173", "Contains parentheses and spaces"),
            ("+123456789", "Too short for pattern - only 9 digits total"),
        ]

        for invalid_phone, description in invalid_phones:
            with pytest.raises(ValidationError):
                Business(
                    name="Acme Corp",
                    email="info@acmecorp.com",
                    phone=invalid_phone,
                    address="123 Business Park Dr",
                    website="https://www.acmecorp.com",
                    industry="Technology",
                )

    def test_empty_phone_converted_to_none(self):
        # Empty phone strings should be converted to None (allowed)
        business = Business(
            name="Acme Corp",
            email="info@acmecorp.com",
            phone="",  # Empty string should become None
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business.phone is None

    def test_phone_with_letters_raises_error(self):
        # Phone numbers with letters should raise ValidationError
        with pytest.raises(ValidationError):
            Business(
                name="Acme Corp",
                email="info@acmecorp.com",
                phone="+1abc2025550173def",
                address="123 Business Park Dr",
                website="https://www.acmecorp.com",
                industry="Technology",
            )

    def test_various_valid_phone_formats(self):
        # E.164 regex validation accepts properly formatted E.164 numbers
        valid_phones = [
            "+12025550173",
            "+442079460958",
            "+33123456789",
            "+81312345678",
            "+61234567890",  # Australia
            "+49301234567",  # Germany
        ]

        for valid_phone in valid_phones:
            business = Business(
                name="Acme Corp",
                email="info@acmecorp.com",
                phone=valid_phone,
                address="123 Business Park Dr",
                website="https://www.acmecorp.com",
                industry="Technology",
            )
            # Phone should be in E.164 format
            assert business.phone is not None
            assert business.phone.startswith("+")
            assert business.phone[1:].isdigit()

    def test_empty_business_name_converted_to_none(self):
        # Empty business names are converted to None (this is acceptable)
        business_empty = Business(
            name="",
            email="info@acmecorp.com",
            phone="+12025550173",
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business_empty.name is None

        business_spaces = Business(
            name="   ",
            email="info@acmecorp.com",
            phone="+12025550173",
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business_spaces.name is None

    def test_phone_boundary_lengths(self):
        # Short valid phone numbers
        business_short = Business(
            name="Acme Corp",
            email="info@acmecorp.com",
            phone="+12025550173",  # US number (11 digits total)
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business_short.phone == "+12025550173"

        # Long valid phone numbers
        business_long = Business(
            name="Acme Corp",
            email="info@acmecorp.com",
            phone="+4915123456789",  # German mobile (13 digits total)
            address="123 Business Park Dr",
            website="https://www.acmecorp.com",
            industry="Technology",
        )
        assert business_long.phone == "+4915123456789"

    def test_phone_boundary_invalid_raise_errors(self):
        # Test some clearly invalid numbers raise ValidationError
        invalid_numbers = [
            "+123456789",  # Too short (9 digits total, pattern needs 10-16)
            "+12345678901234567",  # Too long (17 digits total, pattern allows max 16)
            "+0123456789",  # Invalid: can't start with 0 after +
        ]

        for invalid_phone in invalid_numbers:
            with pytest.raises(ValidationError):
                Business(
                    name="Acme Corp",
                    email="info@acmecorp.com",
                    phone=invalid_phone,
                    address="123 Business Park Dr",
                    website="https://www.acmecorp.com",
                    industry="Technology",
                )

    def test_various_business_types(self):
        # Test different business information
        business_types = [
            ("Tech Startup Inc", "contact@techstartup.io", "Technology"),
            ("Main Street Bakery", "orders@mainstreetbakery.com", "Food & Beverage"),
            ("Johnson & Associates", "info@johnsonlaw.com", "Legal Services"),
            ("Green Energy Solutions", "hello@greenenergy.org", "Renewable Energy"),
        ]

        for name, email, industry in business_types:
            business = Business(
                name=name,
                email=email,
                phone="+12025550173",
                address="123 Business St",
                website="https://www.example.com",
                industry=industry,
            )
            assert business.name == name
            assert business.email is not None
            assert business.email.endswith(
                email.split("@")[1].lower()
            )  # Domain is normalized to lowercase
            assert business.industry == industry

    def test_none_field_handling(self):
        # Test that None can be used for any field
        business_partial = Business(
            name="Incomplete Business",
            email=None,
            phone="+12025550173",
            address=None,
            website="https://www.example.com",
            industry="Unknown",
        )

        assert business_partial.name == "Incomplete Business"
        assert business_partial.email is None
        assert business_partial.phone == "+12025550173"
        assert business_partial.address is None
        assert business_partial.website == "https://www.example.com"
        assert business_partial.industry == "Unknown"

    def test_all_fields_none(self):
        # Test business with all fields as None
        business_all_none = Business(
            name=None, email=None, phone=None, address=None, website=None, industry=None
        )

        assert business_all_none.name is None
        assert business_all_none.email is None
        assert business_all_none.phone is None
        assert business_all_none.address is None
        assert business_all_none.website is None
        assert business_all_none.industry is None

    def test_string_missing_conversion_to_none(self):
        # Test that string 'MISSING' is converted to None
        business = Business(  # type: ignore
            name="Test Business",
            # email missing
            # phone missing
            # address missing
            website="https://test.com",
            industry="Testing",
        )

        assert business.email is None
        assert business.phone is None
        assert business.address is None
        assert business.website == "https://test.com"
        assert business.industry == "Testing"
