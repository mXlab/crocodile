export default async function Home() {
  const files = await fetch("/api/files", { next: { revalidate: 1 } });
  console.log(files);
  return <div>Hello World !</div>;
}
