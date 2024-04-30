## Instructions

Clone the repository

Get the AirBnB New Users Dataset from Kaggle

Unzip the files

<b>--!! The .csv files may be zips at this point !!--</b>

Unzip each of them and put them in the folder ```./raw_data/``` in whichever folder you have cloned the repository

### Alternative Extract with zipped folder already in repo
Use the commands below:

```
mkdir ./raw_data/
tar -xvzf airbnb-recruiting-new-user-bookings.zip -C ./raw_data/
cd ./raw_data/
tar -xzf .\age_gender_bkts.csv.zip
tar -xzf .\countries.csv.zip
tar -xzf .\sample_submission_NDF.csv.zip
tar -xzf .\sessions.csv.zip
tar -xzf .\test_users.csv.zip
tar -xzf .\train_users_2.csv.zip
```

The code should now be executable
