-- Latest Companion phone report, alongside the existing contact timezone.
ALTER TABLE contacts ADD COLUMN location_city TEXT;
ALTER TABLE contacts ADD COLUMN location_region TEXT;
ALTER TABLE contacts ADD COLUMN location_country_code TEXT;
ALTER TABLE contacts ADD COLUMN location_reported_at TEXT;
