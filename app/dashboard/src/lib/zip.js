import JSZip from "jszip";

/**
 * Expands any .zip files in `files` into their contained entries (matching
 * `extensions`) as real File objects, leaving non-zip files untouched — so
 * a zip upload flows through the existing "one row per File" pipeline
 * exactly like dropping its contents individually, no special-casing
 * needed downstream.
 * @param {File[]} files
 * @param {{extensions?: string[]}} [opts]
 * @returns {Promise<File[]>}
 */
export async function expandZipFiles(files, opts = {}) {
  const { extensions = [".csv"] } = opts;
  const out = [];
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      out.push(file);
      continue;
    }
    const zip = await JSZip.loadAsync(file);
    const entries = Object.values(zip.files).filter(
      (e) => !e.dir && extensions.some((ext) => e.name.toLowerCase().endsWith(ext))
    );
    for (const entry of entries) {
      const blob = await entry.async("blob");
      // Zip entries can carry folder paths (e.g. "Train/Train1.csv") — keep
      // just the base filename so it matches Train_Labels.csv's plain names.
      const baseName = entry.name.split("/").pop();
      out.push(new File([blob], baseName, { type: "text/csv" }));
    }
  }
  return out;
}
