[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FN98O3k7)
## Environment setup (Python 3.10+ recommended)
- Clone repo
- Open VSCode and go to the program folder
- Create a venv `python -m venv .venv`
- Enter the venv `.\.venv\Scripts\activate`
- Run `pip install -r requirements.txt`

## Lichess connection
- Create an account on [Lichess.org](https://lichess.org/signup?referrer=https%3A%2F%2Flichess.org%2F)
- Hover over your name in top right, then select `Preferences`->`API access tokens`->`Click the blue button top right to create a token, give your token a name (ex: lichess-bot-token), select all the permissions then click Create.
- Save this token to the config.yml under the token key. **Note to set all the option values of the bot to Green to ensure maximum compatibility. Remember to save your token and do NOT share it or hardcode it in your code.**

![lichess_pic_2](https://github.com/user-attachments/assets/03492f8e-0ae9-495d-9058-f14cc835c82a)
![lichess_pic_3](https://github.com/user-attachments/assets/db262216-1559-4bbc-ac5e-39b83699bda6)

You can run a quick smoke test in a sample python file as in:

From the command line:

```bash
export lichess_token="YOUR_TOKEN"
```
(in Linux/macOS)

```powershell
set lichess_token="YOUR_TOKEN"
```
(in Windows)

In a test_token.py file run:
```
TOKEN = os.environ["lichess_token"]
H = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/x-ndjson",
}

# Smoke Test for
def smoke_test_token():
    global TOKEN
    H = {"Authorization": f"Bearer {TOKEN}"}
    me = requests.get("https://lichess.org/api/account", headers=H).json()
    print("Logged in as:", me["username"])

smoke_test_token()
```

## Upgrade to bot account
After entering your token in the config.yml file, you can run the following command to upgrade your account to a bot account:
```bash
python lichess_bot.py -u
```

If successful, this command will also start running your bot on lichess. Navigate to lichess.org. You should see a robot icon at the top left corner indicating that you are logged in as a bot. Press play against computer and run your bot in a test game, ensuring it is able to make moves. Your starting bot is a random move bot, so don't expect it to play well!

## Customize your bot
Afterwards see the homemade.py file for sample chess engine classes. A template class called MyBot is provided for you to customize your own bot logic. Expand on the search method to implement your own chess engine logic. You may add any functions you need to the class.

The only file you should change is homemade.py. The other files are driver code that connects to lichess and handles all the API calls. The homemade.py file contains the bot class that you can customize.

## Run your bot for testing
To run your bot, simply execute the following command:
```bash
python lichess_bot.py
```

Or 

```bash
python lichess_bot.py -v
```

If you want to see verbose logging output.

## Running your bot during the tournament

During tournament time we will be using the lichess GUI to send match challenges.

## Algorithms You Should Check Out
### Minimax Search
- Core game-tree algorithm: alternate maximizing (your move) and minimizing (opponent) to a fixed depth, then evaluate the leaf with a heuristic.
- Benefits: simple and correct baseline; yields a principal variation (best line) you can display; foundation for all other improvements.

### Alpha-Beta Pruning and Move Ordering
- Alpha-beta keeps best known bounds and prunes branches that cannot influence the final choice; same result as minimax but visits far fewer nodes.
- Good move ordering (e.g., PV move first, captures/checks first, killer/history heuristics) increases pruning effectiveness dramatically.
- Benefits: large speedups (often 10x+), allowing deeper search within the same time budget.

### Capture Chains
- Also called quiescence search: when the frontier is "noisy", extend the search along forcing moves (captures, checks, promotions) until the position becomes quiet.
- Benefits: reduces horizon effects and tactical blunders; stabilizes evaluation in sharp positions.

### Iterative Deepening
- Search depth 1, then 2, etc., until time runs out; each iteration uses results from the previous one for better move ordering.
- Benefits: natural time control (you can stop anytime with the current best move), improved ordering via the previous principal variation, and better responsiveness.

### Transposition Tables
- Cache evaluated positions using a Zobrist hash; store score, depth, node type (exact/alpha/beta), and best move.
- Reuse cached results when the same position is reached via different move orders, and try the TT move first to improve ordering.
- Benefits: avoids re-searching repeated positions, increases pruning, and accelerates deep searches.

### Piece Square Tables
- Simple evaluation technique: assign values to pieces based on their type and position on the board using pre-defined tables.
- Benefits: fast and effective way to capture positional nuances without complex evaluation functions.

## Resources
- Great video on chess engine development that covers some of the above algorithms in detail: https://www.youtube.com/watch?v=U4ogK0MIzqk&t=1008s (Note you can skip the parts that cover board representation and move generation since those are already implemented for you in this starter code).
- Check out the documentation for the python-chess library used in this starter code: https://python-chess.readthedocs.io/en/v1.11.2/ . This will give you a sense of what functions are available for you to use in your bot implementation. The python chess library board representation has lots of helper functions for evaluating board state, for example if pieces are attacking each other, generating legal moves, etc.

## Citation
If this software has been used for research purposes, please cite it using the "Cite this repository" menu on the right sidebar. For more information, check the [CITATION file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/CITATION.cff).
