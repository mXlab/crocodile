"use client";
import { Model } from "@prisma/client";
import { useRouter } from "next/navigation";
import { type } from "os";
import { createContext, useCallback, useState } from "react";

export function RefreshButton() {
  const router = useRouter();
  const handleClick = useCallback(() => {
    router.refresh();
  }, [router]);

  return (
    <button className="btn" onClick={handleClick}>
      Refresh Page
    </button>
  );
}

interface ModelTableColumnProps {
  name: string;
  onClick?: () => void;
}

export function ModelTableColumn({ name, onClick }: ModelTableColumnProps) {
  return (
    <th onClick={onClick} className="cursor-pointer">
      {name}
    </th>
  );
}

interface ModelTableRow {
  model: Model;
}

function ModelTableRow({ model }: ModelTableRow) {
  return (
    <tr>
      <td>{model.id}</td>
      <td>{model.name}</td>
      <td>{model.type}</td>
      <td>{model.created_at.toISOString()}</td>
      <td>{model.iteration}</td>
      <td>{model.fid}</td>
    </tr>
  );
}

interface ModelTableProps {
  models: Model[];
}

export function ModelTable({ models }: ModelTableProps) {
  const [orderedModels, setOrderedModels] = useState(models);
  const handleClick = useCallback(
    (name: string) => {
      const orderedModels = models.toSorted((a: Model, b: Model) => {
        if (typeof a[name] === "string" && typeof b[name] === "string") {
          const nameA = a.name.toUpperCase(); // ignore upper and lowercase
          const nameB = b.name.toUpperCase(); // ignore upper and lowercase
          if (nameA < nameB) {
            return -1;
          }
          if (nameA > nameB) {
            return 1;
          }

          // names must be equal
          return 0;
        } else {
          return a[name] - b[name];
        }
      });
      setOrderedModels(orderedModels);
    },
    [models]
  );

  return (
    <div className="overflow-x-auto">
      <table className="table">
        {/* head */}
        <thead>
          <tr>
            <ModelTableColumn name="Id" onClick={() => handleClick("id")} />
            <ModelTableColumn name="Name" onClick={() => handleClick("name")} />
            <ModelTableColumn name="Type" onClick={() => handleClick("type")} />
            <ModelTableColumn
              name="Created at"
              onClick={() => handleClick("created_at")}
            />
            <ModelTableColumn
              name="Iteration"
              onClick={() => handleClick("iteration")}
            />
            <ModelTableColumn name="FID" onClick={() => handleClick("fid")} />
          </tr>
        </thead>
        <tbody>
          {orderedModels.map((model) => (
            <ModelTableRow key={model.id} model={model} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
