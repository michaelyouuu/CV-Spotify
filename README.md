## CV-Spotify

Hand gesturing spotify volume and playback control

Steps needed:

1. Clone the repository
   
2. pip3 install -r requirements.txt

3. Shift + command + P to open the VSCode Palette and select the correct Python Interpreter (local 3.11/12 should work), if the imports are resolved, there should be no problems underlined in the script. 

4. Create a Spotify Developer App so that it will authorize your Spotify account and recognize playback.
5. a) Visit https://developer.spotify.com/dashboard
   
Create an app to get:

	•	SPOTIPY_CLIENT_ID

	•	SPOTIPY_CLIENT_SECRET

	•	SPOTIPY_REDIRECT_URI

 These should be under settings under your profile. 

 If you are prompted for a website callback URL, use the below URL under redirect.

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

Do _NOTE_ :

Spotify must be open and playing (any song, any volume) when you start running the script. 

You should also disconnect your phone from continuity camera on Facetime if it starts the webcam from your phone. I think you may also be able to just disconnect bluetooth on your phone. If that does not work, manually select your computer webcam at the top of your screen (should be a green camera icon). 
