import numpy as np
import pandas as pd 


class StockAccount:
    
    def __init__(self, num_stock):

        self.num_stock = num_stock
        self.last_action = None


    def initial_account(self, ):
        """
        Initialize the stock account with given account information.
        """
        self.account = {
            "balance": 600000,  # the average 
            "holdings": 0,      # List to hold stock holdings
            "profit": 0,        # Total profit from trades
        }
        self.transaction_cost = 0.003
        self.initial_balance = 600000
    
    
    def update_account(self, action, open_price, close_price):
        """
        Update the stock account with new trade information, 
        include the balance, holdings, and trades.
        """

        if action == 1:
            # buy stock 
            self.account["holdings"] = max(self.account["holdings"], (1 - self.transaction_cost) * self.account['balance'] / open_price)
            self.account["balance"] = 0
            self.account["profit"] = ((self.account["balance"] + close_price * self.account["holdings"])-self.initial_balance) / self.initial_balance

        if action == -1:
            # sell stock 
            self.account["balance"] = max(self.account["balance"], open_price * self.account["holdings"] * (1 - self.transaction_cost))
            self.account["holdings"] = 0
            self.account["profit"] = ((self.account["balance"] + close_price * self.account["holdings"])-self.initial_balance) / self.initial_balance

        if action == 0:
            # hold share or balance
            self.account["profit"] = ((self.account["balance"] + close_price * self.account["holdings"])-self.initial_balance) / self.initial_balance
        
        return self.account

if __name__ == '__main__':

    finance_account = StockAccount()
    


        
