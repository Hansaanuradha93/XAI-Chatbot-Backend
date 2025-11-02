import os
from supabase import create_client, Client

# Supabase Client Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connected.")
else:
    print("⚠️ Supabase credentials not set. Logging disabled.")
