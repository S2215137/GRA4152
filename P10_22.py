## An appointment class for appointments.
class Appointment():
    # Class variable is created since all objects of Appointment superclass needs to communicate
    # with each other.
    # Inputs: 'description' is the description of the appointment, and 'date' is the date of the
    # appointment. The date format is 'dd-mm-yyyy'
    all_appointments = dict()
    # occursOn checks if the appointment occurs on that day.
    def occursOn(self, year: str, month: str, day: str):
        appointments_for_the_method = list()
        try:
            # Sees if it's a one time appointment.
            for i in Appointment.all_appointments['onetime_appointments']:
                if i[1] == f"{day}-{month}-{year}":
                    appointments_for_the_method.append(i[0])
                          
            # Sees if there are any monthly appointments.
        except:
            pass
        # Checking to see for monthly appointments.
        try:
            for i in Appointment.all_appointments['monthly_appointments']:
                if i[1] == str(day):
                    appointments_for_the_method.append(i[0])
        except:
            pass
        # See if there are any daily appointments.
        try:
            for daily_appointment in Appointment.all_appointments['daily_appointments']:
                appointments_for_the_method.append(daily_appointment)

        except:
            pass
        if len(appointments_for_the_method) == 0:
            return print(f"There are no appointments for {day}-{month}-{year}, yet.")

        else:
            print(f"All the appointments for {day}-{month}-{year} is:")
            for appointment in appointments_for_the_method:
                print(appointment)


## The class Onetime represents a one-time appointment.
# This class takes in two inputs, 'description' is the description of the appointment, whereas
# 'date' takes in a date in the format: dd-mm-yyyy.
class Onetime(Appointment):
    def __init__(self, description, date):
        if 'onetime_appointments' in Appointment.all_appointments:
            Appointment.all_appointments['onetime_appointments'].append([description, date])
        else:
            Appointment.all_appointments['onetime_appointments'] = [[description, date]]


## The Monthly class is appointments that occurs one time each month.
class Monthly(Appointment):
    # Inputs are a description of the appointment, and what day of the month it occurs.
    def __init__(self, description, day_of_month):
        # The data structure in this subclass is a dictionary that is added to the class variable, where
        # the key is monthly_appointments and the value is a list containing both the description the day.

        # Checking to see if there are any prior monthly appointments in the object.
        if 'monthly_appointments' in Appointment.all_appointments:
            Appointment.all_appointments['monthly_appointments'].append([description, day_of_month])
        # If this is the first monthly appointment, a key with a list is created and inside the list is the description
        # of the appointment.
        else:
            Appointment.all_appointments['monthly_appointments'] = [[description, day_of_month]]


## The Daily class is for daily appointments.
class Daily(Appointment):
    def __init__(self, description):
        # Checking if there's already been created a daily appointment key
        if 'daily_appointments' in Appointment.all_appointments:
            Appointment.all_appointments['daily_appointments'].append(description)
        # If not, a key with a list is created and inside the list is the description
        # of the appointment.
        else:
            Appointment.all_appointments['daily_appointments'] = [description]


# Instantiating an object from Appointment class, not necessary, but easier for the readability
# of the code.
my_appointments = Appointment()

# Adding several appointments as a part of the prompt.
dmv_appo = Onetime('Appointment with the Norwegian DMV', '12-02-2022')
exercise_appo = Daily('Walk my dog.')
eat_appo = Daily('I have to eat dinner')
hom_deadlines = Monthly('Homework deadline', '20')
exam_day = Onetime('I have an exam today', '12-02-2022')

# Checking to see if there are any appointments that occurs on the 12th of December 2022.
print('expected values:')
print('Walk my dog.', 'I have to eat dinner')
print(my_appointments.occursOn('2022', '12', '12'))