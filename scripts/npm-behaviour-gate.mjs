// Binding behaviour gate: the published bindings answer with the documented
// values and refusals, not just load.
//
//   PKG_NAME=<package-name> node scripts/npm-behaviour-gate.mjs
//
// Run it from a directory where the package is installed -- the publish
// workflow's smoke directory, or any directory with a local build under
// node_modules. The package is resolved from the working directory, not from
// this file. The publish workflow runs it after the README examples and before
// `npm publish`; keeping the assertions in a file rather than in the workflow
// means the same checks can run against a local build before the push that
// publishes, so an assertion the engine no longer satisfies is found there.

import { createRequire } from 'node:module'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

const pkg = process.env.PKG_NAME
if (!pkg) {
  console.error('usage: PKG_NAME=<package-name> node npm-behaviour-gate.mjs')
  process.exit(2)
}
const requireFromCwd = createRequire(join(process.cwd(), 'noop.js'))
const m = requireFromCwd(pkg)
let checks = 0
const ok = (c, msg) => { checks++; if (!c) { console.error('FAIL: ' + msg); process.exit(1) } }

// A sample inside a 0.1 % band with a short upper tail: the profile
// likelihood is monotone across any symmetric range, so the estimate
// belongs on the lower end of it and must say so. Computed from the
// unnormalised transform the likelihood is instead noise at the 1e-5
// level, and the search settles just inside the bound and calls that
// an interior optimum -- which is what a caller cannot tell apart.
const narrow = Array.from({ length: 57 }, (_, i) => 25 + ((i * 7) % 13 - 6) * 0.002)
  .concat([25.021, 25.023, 25.025])
for (const [lo, hi] of [[-5, 5], [-20, 20]]) {
  const e = m.estimate_lambda(narrow, lo, hi)
  ok(e.at_bound === true, `estimate_lambda(narrow, ${lo}, ${hi}).at_bound=${e.at_bound}`)
  ok(e.lambda === lo, `estimate_lambda(narrow, ${lo}, ${hi}).lambda=${e.lambda}`)
}

// An interior optimum stays interior: the endpoint comparison must
// not turn every estimate into a bound.
const expo = Array.from({ length: 30 }, (_, i) => Math.exp((i + 1) * 0.2))
const e = m.estimate_lambda(expo, -2, 2)
ok(e.at_bound === false, `exponential sample reported at_bound=${e.at_bound}`)
ok(Math.abs(e.lambda) < 0.3, `exponential sample lambda=${e.lambda}`)

// box_cox keeps its meaning where the old expression could still read it.
const bc = m.box_cox([1, Math.E], 0)
ok(Math.abs(bc[0]) < 1e-12 && Math.abs(bc[1] - 1) < 1e-9, `box_cox lambda=0 gave ${bc}`)

// Critical values against the NIST/SEMATECH tables, including df = 1,
// where a Newton step from the normal quantile has the farthest to go.
for (const [p, df, want] of [[0.975, 1, 12.7062], [0.975, 10, 2.2281], [0.95, 8, 1.8595]]) {
  const t = m.t_distribution_quantile(p, df)
  ok(Math.abs(t - want) < 5e-4, `t_distribution_quantile(${p}, ${df})=${t}, want ${want}`)
}
const c2 = m.chi_squared_quantile(0.975, 2)
ok(Math.abs(c2 - 7.3778) < 1e-3, `chi_squared_quantile(0.975, 2)=${c2}`)
// An argument outside the domain throws instead of returning NaN.
let threw = false
try { m.t_distribution_quantile(1, 5) } catch { threw = true }
ok(threw, 't_distribution_quantile(1, 5) did not throw')

// A data array is read as sent: null is refused where it sits, not
// averaged as 0 by the glue's typed-array copy.
let refused = null
try { m.mean([1, null, 3]) } catch (err) { refused = err }
ok(refused && refused.code === 'malformed_input' && refused.index === 1,
  `mean([1, null, 3]) gave ${refused ? refused.code + '@' + refused.index : 'no error'}`)
ok(m.mean(new Float64Array([1, 2, 3])) === 2, 'mean(Float64Array) changed')

console.log(`binding behaviour gate: ok (${checks} checks)`)
