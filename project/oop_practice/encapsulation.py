# Using Getter and Setter Methods
# Encapsulation often involves controlling access to attributes using getter and setter methods. These methods allow indirect interaction with private or protected attributes.



# Encapsulation example

class BankAccount:
    def __init__(self, account_holder, balance):
        self.account_holder = account_holder  # Public attribute
        self.__balance = balance  # Private attribute (indicated by the double underscore)

    # Public method to check the balance
    def get_balance(self):
        return self.__balance

    # Public method to deposit money
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"{amount} deposited. New balance is {self.__balance}.")
        else:
            print("Deposit amount must be positive!")

    # Public method to withdraw money
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"{amount} withdrawn. Remaining balance is {self.__balance}.")
        else:
            print("Invalid withdrawal amount!")


if __name__ == "__main__":

    # Creating an object
    account = BankAccount("Mark", 1000)

    # Accessing public attribute
    print(f"Account holder: {account.account_holder}")

    # Accessing private attribute directly (this will raise an error)
    # print(account.__balance)  # Uncommenting this line will cause an AttributeError

    # Accessing private attribute via public methods
    print(f"Current balance: {account.get_balance()}")

    # Using methods to modify private attribute
    account.deposit(500)
    account.withdraw(300)