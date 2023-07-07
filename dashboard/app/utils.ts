import { watch } from "node:fs";

export function watchDir(dir: string) {
  watch(dir, (eventType, filename) => {
    console.log(`event type is: ${eventType}`);
    if (filename) {
      console.log(`filename provided: ${filename}`);
    } else {
      console.log("filename not provided");
    }
  });
}
