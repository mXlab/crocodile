import { PrismaClient, Model } from "@prisma/client";
import { NextResponse } from "next/server";
import { ModelTable, RefreshButton } from "./components";

const prisma = new PrismaClient();

export const revalidate = 1;

export default async function Home() {
  const models = await prisma.model.findMany();
  return (
    <div>
      <h1>Hello World !</h1>
      <RefreshButton />
      <ModelTable models={models} />
    </div>
  );
}
