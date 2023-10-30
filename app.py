import os
import sqlite3
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import apology, login_required, lookup, usd
import yfinance as yf
import plotly.express as px
from datetime import datetime
from time import sleep
from decision_creation import stock_trading_inference
from Model_Traningt import traning


# Configure application
app = Flask(__name__)



def Replace(word):
    replace_word = str(word).replace('(', '').replace(')','').replace(',', '').replace("'", "")
    return replace_word


# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
con = sqlite3.connect("instance\\finance1.db", check_same_thread=False)
db = con.cursor()

# Create tables if they don't exist
db.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        hash TEXT NOT NULL,
        cash NUMERIC DEFAULT 10000.00
    )
""")

db.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        shares INTEGER NOT NULL,
        price NUMERIC NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
""")


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Function to get the user's cash balance


def get_cash_balance(user_id):
    cas = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
    cash = cas.fetchone()
    cash = Replace(cash)
    return float(cash) if cash else 0


@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""
    user_id = session["user_id"]

    # Get the user's cash balance
    ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
    cash = ca.fetchone()

    # Query user's stock holdings
    ro = db.execute("""
        SELECT symbol, SUM(shares) AS total_shares
        FROM transactions
        WHERE user_id = ?
        GROUP BY symbol
        HAVING total_shares > 0
    """, (user_id,))
    rows = ro.fetchall()
    # print(cash,rows)
    if rows:

        # Initialize variables to store portfolio data
        cash = str(cash[0])
        cash = float(cash)

        # print(cash)
        # print(type(cash))
        stocks = []
        total_portfolio_value = cash

        # Iterate through the list of stocks
        for row in rows:
            stock_symbol = row[0]
            total_shares = row[1]
            stock_info = lookup(stock_symbol)

            if stock_info:
                stock_name = stock_info["name"]
                stock_price = stock_info["price"]
                total_value = stock_price * total_shares

                # Append stock data to the list
                stocks.append({
                    "symbol": stock_symbol,
                    "name": stock_name,
                    "shares": total_shares,
                    "price": usd(stock_price),
                    "total_value": usd(total_value)
                })

                # Calculate the overall portfolio value
                total_portfolio_value += total_value

        return render_template("index.html", stocks=stocks, cash=usd(cash), total=usd(total_portfolio_value))
    return redirect('/buy')


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":
        user_id = session["user_id"]
        symbol = request.form.get("symbol")
        shares_str = request.form.get("shares")

        # Check if shares_str can be converted to a float (fractional shares)
        try:
            shares = float(shares_str)
        except ValueError:
            return apology("Invalid number of shares", 400)

        # Check if shares is negative or zero
        if shares <= 0:
            return apology("Number of shares must be positive", 400)

        if not symbol:
            return apology("Must provide stock symbol", 400)

        stock_info = lookup(symbol)

        if not stock_info:
            return apology("Invalid stock symbol", 400)

        cash = get_cash_balance(user_id)
        total_cost = stock_info["price"] * shares

        if cash < total_cost:
            return apology("Insufficient funds", 400)

        # Record the purchase in the transactions table
        db.execute("INSERT INTO transactions (user_id, symbol, shares, price) VALUES (?, ?, ?, ?)",
                   (user_id, symbol, shares, stock_info["price"]))
        con.commit()

        # Update user's cash balance
        db.execute("UPDATE users SET cash = cash - ? WHERE id = ?",
                   (total_cost, user_id))
        con.commit()

        flash("Purchase successful!")

        return redirect("/")

    else:
        user_id = session["user_id"]
        ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
        original_value =float(Replace(str(ca.fetchone())))
        cash = round(original_value, 2)
        return render_template("buy.html",cash=cash)


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    user_id = session["user_id"]
    transactions=[]

    # Query user's transaction history
    transaction = db.execute("""
        SELECT symbol, shares, price, timestamp
        FROM transactions
        WHERE user_id = ?
        ORDER BY timestamp DESC
    """, (user_id,))
    transaction = transaction.fetchall()
    for i in transaction:
        timestamp=i[3]
        symbol=i[0]
        shares=i[1]
        price=i[2]

        transactions.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "shares": shares,
            "price": price,
            
        })
                    # Get the user's cash balance
    ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
    original_value =float(Replace(str(ca.fetchone())))
    cash = round(original_value, 2)

    return render_template("history.html", transactions=transactions,cash=Replace(cash))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    session.clear()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Ensure username was submitted+
        if not username:
            return apology("must provide username", 403)

        # Ensure password was submitted
        if not password:
            return apology("must provide password", 403)

        # Query database for username
        try:
            rows = db.execute(
                "SELECT * FROM users WHERE username = ?", (username,))
            data = rows.fetchone()
            # ca=data[1]
            # print(data)

            # # Check if the username exists and the password is correct
            if check_password_hash(data[2], password) != True:
                return apology("invalid username and/or password", 403)

            # Remembering which user has logged in
            session["user_id"] = data[0]

            # Redirect user to home page
            return redirect("/")
        except:
            return apology("User not Exist", 403)

    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""
    # Forget any user_id
    session.clear()
    # Redirect user to login form
    # db.close()
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    try:
        if request.method == "POST":
            user_id = session["user_id"]
            while True:
                
                # Get the stock symbol entered by the user
                symbol = request.form.get("symbol")
                
                

                # Get the user's cash balance
                ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
                cash = ca.fetchone()
                # Using the lookup function to get stock information
                stock_info = lookup(symbol)
                live_data = yf.download(symbol, period="1d",interval="2m")
                datas=str(stock_trading_inference(symbol,Replace(cash)))
                dynamic_data = [['Year', f' Time to {datas} ']]

                for index, row in live_data.iterrows():
                    original_datetime = str(index)
                    # Parse the datetime string
                    parsed_datetime = datetime.fromisoformat(original_datetime)
                    # Extract time and timezone
                    time = parsed_datetime.strftime('%H:%M:%S')
                    closing_price = row['Close']
                    dynamic_data.append([str(time), closing_price])

                if stock_info:
                    # Rendering the quoted.html template with stock information
                    user_id = session["user_id"]
                    ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
                    original_value =float(Replace(str(ca.fetchone())))
                    cash = round(original_value, 2)
                    return render_template("quoted.html",stock=stock_info,data=dynamic_data,cash=cash)
                sleep(10)

        else:
            # Rendering the quote.html template for GET requests
                        # Get the user's cash balance
            user_id = session["user_id"]
            ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
            original_value =float(Replace(str(ca.fetchone())))
            cash = round(original_value, 2)
            return render_template("quote.html",cash=cash)
    except:
                    # Get the user's cash balance
        rror="refrace"
        return render_template("quote.html",cash=rror) 
        
    


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        # Ensure username was submitted and not blank
        if not username or username.strip() == "":
            return apology("must provide username", 400)

        # Ensure password was submitted
        if not password:
            return apology("must provide password", 400)

        # Ensure password and confirmation match
        if password != confirmation:
            return apology("passwords do not match", 400)

        # Check if username already exists
        existing_user = db.execute(
            "SELECT id FROM users WHERE username = ?", (username,))
        es = existing_user.fetchone()
        # print(es)
        if es:
            return apology("username already exists", 400)

        # Insert new user into the database
        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)",
                   (username, generate_password_hash(password)))
        con.commit()

        flash("Registration successful! Please log in.")
        sleep(2)
        return redirect("/login")

    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    if request.method == "POST":
        user_id = session["user_id"]
        symbol = Replace(request.form.get("symbol"))
        shares = int(request.form.get("shares"))

        if not symbol:
            return apology("must provide stock symbol", 400)
        if not shares or shares <= 0:
            return apology("invalid number of shares", 400)

        stock_info = lookup(symbol)

        if not stock_info:
            return apology("invalid stock symbol", 400)

        # Check if the user owns enough shares to sell
        user_shares = db.execute("""
            SELECT SUM(shares) AS total_shares
            FROM transactions
            WHERE user_id = ? AND symbol = ?
        """, (user_id, symbol))
        user_shares = user_shares.fetchall()
        user_shares = Replace(user_shares)

        if not user_shares or int(user_shares[1]) < shares:
            return apology("insufficient shares to sell", 400)

        # Calculate the total sale value
        total_sale_value = stock_info["price"] * shares

        # Record the sale in the transactions table
        db.execute("INSERT INTO transactions (user_id, symbol, shares, price) VALUES (?, ?, ?, ?)",
                   (user_id, symbol, -shares, stock_info["price"]))
        con.commit()

        # Update user's cash balance
        db.execute("UPDATE users SET cash = cash + ? WHERE id = ?",
                   (total_sale_value, user_id))
        con.commit()
        flash("Sale successful!")
        return redirect("/")

    else:
        user_id = session["user_id"]

        stockss = db.execute("""
            SELECT symbol
            FROM transactions
            WHERE user_id = ?
            GROUP BY symbol
            HAVING SUM(shares) > 0
        """, (user_id,))
        stocks = []
        stockss = stockss.fetchall()
        for stock in stockss:
            s = Replace(stock)
            stocks.append(s)
        user_id = session["user_id"]
        ca = db.execute("SELECT cash FROM users WHERE id = ?", (user_id,))
        original_value =float(Replace(str(ca.fetchone())))
        cash = round(original_value, 2)

        return render_template("sell.html", stocks=stocks,cash=cash)



if __name__ == "__main__":
    app.run(debug="True")
    

    




