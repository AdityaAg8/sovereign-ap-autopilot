/** Spec-correct polyfills for the three ES2026 built-ins pdf.js 6's modern build calls but
 * Chromium < 145-class browsers lack: Map/WeakMap.prototype.getOrInsert(Computed) (proposal-upsert)
 * and Math.sumPrecise (proposal-math-sum). Loaded only with the PDF viewer — bundled into its
 * chunk (pdfjs.ts) and imported first by the worker entry (pdf.worker.entry.mjs) — and installs
 * nothing where the browser already has them. Plain JS so the worker can import it unbundled. */

function define(target, name, value) {
  if (typeof target[name] === "function") return;
  Object.defineProperty(target, name, { value, writable: true, configurable: true });
}

function canBeHeldWeakly(value) {
  if (value !== null && (typeof value === "object" || typeof value === "function")) return true;
  return typeof value === "symbol" && Symbol.keyFor(value) === undefined;
}

// proposal-upsert: Map.prototype.getOrInsert / getOrInsertComputed.
{
  const { get, has, set } = Map.prototype;
  define(Map.prototype, "getOrInsert", function getOrInsert(key, value) {
    if (has.call(this, key)) return get.call(this, key);
    set.call(this, key, value);
    return value;
  });
  define(Map.prototype, "getOrInsertComputed", function getOrInsertComputed(key, callbackfn) {
    has.call(this, key); // brand check first: throws on a non-Map receiver
    if (typeof callbackfn !== "function") throw new TypeError("callbackfn is not a function");
    if (Object.is(key, -0)) key = 0;
    if (has.call(this, key)) return get.call(this, key);
    const value = callbackfn(key);
    // The callback may have inserted the key itself; the computed value overwrites it.
    set.call(this, key, value);
    return value;
  });
}

// proposal-upsert: WeakMap.prototype.getOrInsert / getOrInsertComputed.
{
  const { get, has, set } = WeakMap.prototype;
  define(WeakMap.prototype, "getOrInsert", function getOrInsert(key, value) {
    has.call(this, key);
    if (!canBeHeldWeakly(key)) throw new TypeError("Invalid value used as weak map key");
    if (has.call(this, key)) return get.call(this, key);
    set.call(this, key, value);
    return value;
  });
  define(WeakMap.prototype, "getOrInsertComputed", function getOrInsertComputed(key, callbackfn) {
    has.call(this, key);
    if (typeof callbackfn !== "function") throw new TypeError("callbackfn is not a function");
    if (!canBeHeldWeakly(key)) throw new TypeError("Invalid value used as weak map key");
    if (has.call(this, key)) return get.call(this, key);
    const value = callbackfn(key);
    set.call(this, key, value);
    return value;
  });
}

// proposal-math-sum: Math.sumPrecise — the exact sum of the iterable's numbers, rounded once.
// Shewchuk's exact partials (as in Python's math.fsum), with the proposal's handling of -0,
// infinities, NaN and intermediate overflow; follows the proposal's reference polyfill.
{
  const TWO_1023 = 2 ** 1023;
  const MAX_ULP = 2 ** 971; // the gap between Number.MAX_VALUE and the next-lower double

  // Fast two-sum: exact when |a| >= |b|.
  const twoSum = (a, b) => {
    const hi = a + b;
    return { hi, lo: b - (hi - a) };
  };

  define(Math, "sumPrecise", function sumPrecise(items) {
    // The proposal's first step: a non-Object argument throws, even an iterable primitive ("").
    if (
      items === null ||
      (typeof items !== "object" && typeof items !== "function") ||
      typeof items[Symbol.iterator] !== "function"
    ) {
      throw new TypeError("Math.sumPrecise requires an iterable");
    }
    const partials = [];
    let overflow = 0; // in units of 2 ** 1024
    let state = "minus-zero"; // "minus-zero" | "finite" | "+inf" | "-inf" | "nan"
    let count = 0;

    const iterator = items[Symbol.iterator]();
    for (let step = iterator.next(); !step.done; step = iterator.next()) {
      const value = step.value;
      if (++count >= 2 ** 53) {
        iterator.return?.();
        throw new RangeError("Math.sumPrecise: too many values");
      }
      if (typeof value !== "number") {
        iterator.return?.();
        throw new TypeError("Math.sumPrecise only sums numbers");
      }
      if (state === "nan") continue;
      if (Number.isNaN(value)) state = "nan";
      else if (value === Infinity) state = state === "-inf" ? "nan" : "+inf";
      else if (value === -Infinity) state = state === "+inf" ? "nan" : "-inf";
      else if ((state === "minus-zero" || state === "finite") && !Object.is(value, -0)) {
        state = "finite";
        let x = value;
        let used = 0;
        for (let i = 0; i < partials.length; i++) {
          let y = partials[i];
          if (Math.abs(x) < Math.abs(y)) [x, y] = [y, x];
          let { hi, lo } = twoSum(x, y);
          if (Math.abs(hi) === Infinity) {
            const sign = hi === Infinity ? 1 : -1;
            overflow += sign;
            x = x - sign * TWO_1023 - sign * TWO_1023;
            if (Math.abs(x) < Math.abs(y)) [x, y] = [y, x];
            ({ hi, lo } = twoSum(x, y));
          }
          if (lo !== 0) partials[used++] = lo;
          x = hi;
        }
        partials.length = used;
        if (x !== 0) partials.push(x);
      }
    }

    if (state === "nan") return NaN;
    if (state === "+inf") return Infinity;
    if (state === "-inf") return -Infinity;
    if (state === "minus-zero") return -0;

    let n = partials.length - 1;
    let hi = 0;
    let lo = 0;
    if (overflow !== 0) {
      const next = n >= 0 ? partials[n] : 0;
      n--;
      if (Math.abs(overflow) > 1 || (overflow > 0 && next > 0) || (overflow < 0 && next < 0)) {
        return overflow > 0 ? Infinity : -Infinity;
      }
      ({ hi, lo } = twoSum(overflow * TWO_1023, next / 2));
      lo *= 2;
      if (Math.abs(2 * hi) === Infinity) {
        // Only just above the largest double: it rounds down to MAX_VALUE exactly when the
        // remainder is half an ulp below it and the rest of the sum pulls further down.
        if (hi > 0) {
          return hi === TWO_1023 && lo === -(MAX_ULP / 2) && n >= 0 && partials[n] < 0
            ? Number.MAX_VALUE
            : Infinity;
        }
        return hi === -TWO_1023 && lo === MAX_ULP / 2 && n >= 0 && partials[n] > 0
          ? -Number.MAX_VALUE
          : -Infinity;
      }
      if (lo !== 0) {
        partials[n + 1] = lo;
        n++;
        lo = 0;
      }
      hi *= 2;
    }

    while (n >= 0) {
      const x = hi;
      const y = partials[n];
      n--;
      ({ hi, lo } = twoSum(x, y));
      if (lo !== 0) break;
    }
    // Round-half-even correction when the discarded remainder is exactly half an ulp.
    if (n >= 0 && ((lo < 0 && partials[n] < 0) || (lo > 0 && partials[n] > 0))) {
      const y = lo * 2;
      const x = hi + y;
      if (y === x - hi) hi = x;
    }
    return hi;
  });
}
