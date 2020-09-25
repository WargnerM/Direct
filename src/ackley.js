'use strict';

function ackley(x, options = {}) {
  const { a = 20, b = 0.2, c = 2 * Math.PI, d = x.length } = options;

  if (options.a === undefined) {
    options.a = a;
  }
  if (options.b === undefined) {
    options.b = b;
  }
  if (options.c === undefined) {
    options.b = b;
  }
  if (options.d === undefined) {
    options.b = b;
  }

  var sum1 = 0;
  var sum2 = 0;

  for (let i = 0; i < d; i++) {
    sum1 += Math.pow(x[i], 2);
    sum2 += Math.cos(c * x[i]);
  }
  let y1 = -a * Math.exp(-b * Math.sqrt(sum1 / d));
  let y2 = -1 * Math.exp(sum2 / d);
  let y3 = a + Math.exp(1);
  return y1 + y2 + y3;
}
const npoints = 40 * 1024;
let x = new Array(npoints).fill(5).map((a, i) => a + i);


