from faker import Faker

fake = Faker()

if __name__=="__main__":
    print({
        'full_name': fake.name(),
        'email': fake.email(),
        'address': fake.address(),
        'phone': fake.phone_number(),
        'job': fake.job(),
        'company': fake.company(),
        'registered_date': fake.date_this_decade().isoformat()
    })