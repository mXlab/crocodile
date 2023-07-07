# Dashboard App

## Installation

0. (Beluga Only) Load nodejs module: `module load nodejs`
1. Install all the required npm packages: `npm install`
2. Update the `.env` file with the correct path to the SQLite database
3. Initialize the prisma client: `npx prisma generate`

## Running the app

`npm run dev`

## Specific instruction for running the app on Beluga

1. Create a ssh tunnel to Beluga: `ssh -L 3000:localhost:3000 [USERNAME]@beluga.computecanada.ca`
2. Create a screen: `screen -S dashboard` or resume the screen: `screen -r dashboard`
3. Load nodejs module: `module load nodejs`
4. Navigate to the dashboard app directory: `cd [PATH_TO_DASHBOARD_APP]`
5. Launch the app: `npm run dev`
