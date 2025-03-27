## CV-Spotify

Hand gesturing spotify volume and playback control

Steps needed:

1. Clone the repository
   
2. pip install -r requirements.txt

3. Create a Spotify Developer App so that it will authorize your Spotify account and recognize playback.
4. a) Visit https://developer.spotify.com/dashboard
   
Create an app to get:

	•	SPOTIPY_CLIENT_ID

	•	SPOTIPY_CLIENT_SECRET

	•	SPOTIPY_REDIRECT_URI

  b) Create a .env file

Which should look like this:
***
 ----- .env file ----

SPOTIPY_CLIENT_ID=your_client_id

SPOTIPY_CLIENT_SECRET=your_client_secret

SPOTIPY_REDIRECT_URI=http://localhost:8888/callback

***
4. After that, simply run the program main.py with:
   python main.py
