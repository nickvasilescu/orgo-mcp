// Refreshes the bundled snapshot from the published spec, so the offline
// fallback stays close to the live API. Run before cutting a release.

import { writeFile } from 'node:fs/promises'
import { buildTools } from '../src/tools.js'
import { fetchSpec, SNAPSHOT, SPEC_URL } from '../src/spec.js'

const spec = await fetchSpec()
if (!spec) {
  console.error(`Could not fetch ${SPEC_URL}`)
  process.exit(1)
}

await writeFile(SNAPSHOT, `${JSON.stringify(spec, null, 2)}\n`)
console.log(`${SNAPSHOT}: Orgo API ${spec.info.version}, ${buildTools(spec).length} tools`)
