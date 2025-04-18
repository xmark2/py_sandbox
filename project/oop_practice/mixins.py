class LoggingMixin:
    def log(self, message):
        print(f"LOG: {message}")

class Order(LoggingMixin):
    def create_order(self):
        self.log("Order created")

class Payment(LoggingMixin):
    def process_payment(self):
        self.log("Payment processed")

if __name__ == "__main__":
    order = Order()
    order.create_order()  # Outputs: LOG: Order created

    payment = Payment()
    payment.process_payment()  # Outputs: LOG: Payment processed