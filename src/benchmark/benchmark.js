'use strict';

const Benchmark = require('benchmark');
const optimizeGLSum = require('../optimize/optimizeGLSum');

const { sumOfGaussianLorentzians, optimizeGaussianLorentzianSum } = require('ml-optimize-lorentzian')

let nbPoints = 31;
let tFactor = 0.1;
let t = new Float64Array(nbPoints);
for (let i = 0; i < nbPoints; i++) {
  t[i] = (i - nbPoints / 2) * tFactor;
}

let suite = new Benchmark.Suite();

let pTrue = [
  0,
  0,
  0.001,
  0.001,
  0.31,
  0.31,
  (tFactor * nbPoints) / 10,
  (tFactor * nbPoints) / 10,
];

const pFunction = sumOfGaussianLorentzians(pTrue);
const y = pFunction(t);

const group = [
  { x: 0.1, y: 0.0009, width: (tFactor * nbPoints) / 6 },
  { x: 0.1, y: 0.0009, width: (tFactor * nbPoints) / 6 },
]

let direct, lm;
suite
  .add('Direct: gaussian lorentzian sum', function () {
    direct = optimizeGLSum([t, y], group, { iterations: 20, initialState: {} });
  })
  .add('Levenberg-marquardt: gaussian lorentzian sum', function () {
    lm = optimizeGaussianLorentzianSum([t, y], group);
  })
  .on('cycle', function (event) {
    console.log(String(event.target));
  })
  .on('complete', function () {
    console.log(`Fastest is ${this.filter('fastest').map('name')}`);
  })
  .run();
console.log('Real parameters', pTrue);
console.log('Direct', direct);
console.log('Levenberg Marquardt', lm)

