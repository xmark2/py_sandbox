class DynamicAttributes:
    def __getattr__(self, name):
        print(f"Getting {name} dynamically!")
        return None

    def __setattr__(self, name, value):
        print(f"Setting {name} to {value}")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        print(f"Deleting {name}")
        super().__delattr__(name)

if __name__ == "__main__":
    obj = DynamicAttributes()
    obj.new_attr = 42  # Logs: Setting new_attr to 42
    print(obj.new_attr)  # Logs: Getting new_attr dynamically!, Outputs: 42
    # del obj.new_attr  # Logs: Deleting new_attr
    obj.new_attr2 = 3
    print(obj.new_attr2)
    print(dir(obj))
