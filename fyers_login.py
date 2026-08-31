from fyers_apiv3 import fyersModel
from urllib.parse import urlparse, parse_qs
import webbrowser

# --- REPLACE THESE WITH YOUR EXACT FYERS APP DETAILS ---
client_id = "client_id"       
secret_key = "secret_id"      
redirect_uri = "https://127.0.0.1/"
# -------------------------------------------------------

def generate_daily_token():
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )

    auth_url = session.generate_authcode()
    print("Opening browser for Fyers Login...")
    webbrowser.open(auth_url)

    print("\nAfter logging in, you will be redirected to a blank page.")
    print("Copy the ENTIRE URL from your browser's address bar and paste it below.")
    redirected_url = input("\nPaste URL here: ").strip()

    try:
        parsed_url = urlparse(redirected_url)
        auth_code = parse_qs(parsed_url.query)['auth_code'][0]
    except KeyError:
        print("\n❌ Error: Invalid URL pasted. Could not find 'auth_code'.")
        return

    session.set_token(auth_code)
    response = session.generate_token()

    if "access_token" in response:
        print("\n✅ SUCCESS! Copy your Access Token below:")
        print("---------------------------------------------------")
        print(response["access_token"])
        print("---------------------------------------------------")
        print("Paste this into the Streamlit sidebar in trade.py")
    else:
        print("\n❌ Error generating token:", response)

if __name__ == "__main__":
    generate_daily_token()
