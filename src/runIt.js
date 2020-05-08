
const sumOfPseudoVoigt = require('./functions/sumOfPseudoVoigt');
const optimizePseudoVoigtSum = require('./optimize/optimizePseudoVoigtSum');
// const fs = require('fs')
// const path = require('path');
// const { NMR } = require('spectra-data');
// console.log(SD)
// return

// const pathToData = '/home/abolanos/airwaveJcampAdmin';
// const filename = 'Airwave-9-780-1.jdx'

// let file = fs.readFileSync(path.join(pathToData, filename), 'utf8');

// let spectrum = NMR.fromJcamp(file);
// let from = 1.22;
// let to = 1.26;

// let guest = spectrum.getPeaks({from, to, noiseLevel: 0});
// let vector = spectrum.getVector({from, to, outputX: true});
// console.log(vector, guest)
// return
// console.time('Direct')
// let resultado = JSON.stringify(fs.readFileSync('resultado','utf8'));
// let result = optimizePseudoVoigtSum([vector.x, vector.y], guest, {iterations: 55, initialState: resultado.finalState})
// fs.writeFileSync('optimum', JSON.stringify(result.optimum))
// fs.writeFileSync('resultado', JSON.stringify(result))

let nbPoints = 1024;
let minX = 2;
let maxX = 5;

let group = [];
for (let i = 0; i < 50; i++) {
  group.push({
    x: minX + (Math.random() * (maxX - minX)),
    y: 2 + Math.random() * 5,
    width: 0.02 + Math.random() * 0.03,
    xL: Math.random()
  });
}
let nL = group.length;
let pTrue = new Array(group.length * 4);
for (let i = 0; i < nL; i++) {
  pTrue[i] = group[i].x;
  pTrue[i + nL] = group[i].y;
  pTrue[i + 2 * nL] = group[i].width;
  pTrue[i + 3 * nL] = group[i].xL;
  group[i].x += Math.random() * 0.1;
  group[i].y += Math.random() * 0.1;
  group[i].width += Math.random() * 0.01;
}

let x = new Array(nbPoints);
let y = new Array(nbPoints).fill(0);
let jump = (maxX - minX) / (nbPoints - 1);
for (let i = 0; i < nbPoints; i++) {
  x[i] = minX + i * jump;
}
y = sumOfPseudoVoigt(x, pTrue, [0]);

let result = optimizePseudoVoigtSum([x, y], group, { iterations: 10, initialState: {} });
console.log(result.optimum[0], pTrue);


