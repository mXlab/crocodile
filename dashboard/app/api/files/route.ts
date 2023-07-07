import { NextResponse } from "next/server";
import fs from "fs";

export async function GET() {
  if (!process.env.EXPERIMENTS_PATH) {
    throw new Error("EXPERIMENTS_PATH is not set");
  }
  const files = fs.readdirSync(process.env.EXPERIMENTS_PATH);
  console.log(files);
  return NextResponse.json(files);
}
