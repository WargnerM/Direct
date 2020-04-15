const { direct } = require('./index');
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
  })
}
let nL = group.length
let pTrue = new Array(group.length * 4);
for (let i = 0; i < nL; i++) {
  pTrue[i] = group[i].x
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
// console.log(group[0], pTrue[0])
// return

let result = optimizePseudoVoigtSum([x, y], group, {iterations: 100, initialState: {}})
console.log(result.optimum[0], pTrue);

function optimizePseudoVoigtSum(xy, group, options = {}) {
  var {
    percentage = 0,
    iterations = 20,
    noiseLevel = 0,
    initialState = {}
  } = options;

  var xy2 = parseData(xy, percentage);
  if (xy2 === null || xy2[0].rows < 3) {
    return null; //Cannot run an optimization with less than 3 points
  }

  var t = xy2[0];
  var yData = xy2[1];
  var maxY = xy2[2];
  var nbPoints = t.length;
  var consts = [noiseLevel / maxY]; // optional vector of constants
  var nL = group.length;
  var pMin = new Array(nL * 4, 1);
  var pMax = new Array(nL * 4, 1);
  var dt = Math.abs(t[0] - t[1]);

  for (let i = 0; i < nL; i++) {
    pMin[i] = group[i].x - dt;
    pMin[i + nL] = 0;
    pMin[i + 2 * nL] = group[i].width * 0.8;
    pMin[i + 3 * nL] = 0;

    pMax[i] = group[i].x + dt;
    pMax[i + nL] = group[i].y / maxY * 1.1;
    pMax[i + 2 * nL] = group[i].width * 1.2;
    pMax[i + 3 * nL] = 1;
  }

  var targetFunc = getErrorFunction(sumOfPseudoVoigt, {x: t, y: yData}, consts);

  // return
  var result = direct(
    targetFunc,
    pMin,
    pMax,
    {iterations},
    initialState
  );

  
  // //Put back the result in the correct format
  for (let k = 0; k < result.optimum.length; k++) {
      let pFit = result.optimum[k];
      for (let i = 0; i < nL; i++) {
          pFit[i + nL] *= maxY;
      }
      result.optimum[k] = pFit
  }
  return result;
}
function parseData(xy, threshold) {
  var nbSeries = xy.length;
  var t = null;
  var y_data = null,
    x,
    y;
  var maxY = 0,
    i,
    j;

  if (nbSeries == 2) {
    //Looks like row wise matrix [x,y]
    var nbPoints = xy[0].length;
    //if(nbPoints<3)
    //    throw new Exception(nbPoints);
    //else{
    t = new Array(nbPoints); //new Matrix(nbPoints,1);
    y_data = new Array(nbPoints); //new Matrix(nbPoints,1);
    x = xy[0];
    y = xy[1];
    if (typeof x[0] === "number") {
      for (i = 0; i < nbPoints; i++) {
        t[i] = x[i];
        y_data[i] = y[i];
        if (y[i] > maxY) maxY = y[i];
      }
    } else {
      //It is a colum matrix
      if (typeof x[0] === "object") {
        for (i = 0; i < nbPoints; i++) {
          t[i] = x[i];
          y_data[i] = y[i];
          if (y[i] > maxY) maxY = y[i];
        }
      }
    }
  } else {
    var nbPoints = nbSeries;
    t = new Array(nbPoints); //new Matrix(nbPoints, 1);
    y_data = new Array(nbPoints); //new Matrix(nbPoints, 1);
    for (i = 0; i < nbPoints; i++) {
      t[i] = xy[i];
      y_data[i] = xy[i];
      if (y_data[i] > maxY) maxY = y_data[i];
    }
  }
  for (i = 0; i < nbPoints; i++) {
    y_data[i] /= maxY;
  }
  if (threshold) {
    for (i = nbPoints - 1; i >= 0; i--) {
      if (y_data[i] < threshold) {
        y_data.splice(i, 1);
        t.splice(i, 1);
      }
    }
  }
  if (t.length > 0)
    return [
      t,
      y_data,
      maxY
    ];
  return null;
}

function getErrorFunction(func, data, constants) {
  return (params) => {
      let y = func(data.x, params, constants);
      let errorValue = 0;
      for (let i = 0; i < data.x.length; i++) {
          errorValue += y[i] - data.y[i];
      }
      return Math.pow(errorValue, 2);
  }
}

function sumOfPseudoVoigt(t, p, c) {
  var nL = p.length / 4,
    factorG,
    factorL,
    cols = t.length,
    p2;
  var result = new Array(t.length).fill(c[0]);
  // console.log('the first is %s', result[0][0])
  for (let i = 0; i < nL; i++) {
    var xL = p[i + nL * 3];
    var xG = 1 - xL;
    p2 = Math.pow(p[i + nL * 2], 2);
    factorL = xL * p[i + nL] * p2;
    factorG = xG * p[i + nL];
    for (let j = 0; j < cols; j++) {
      result[j] +=
        factorG * Math.exp(-Math.pow(t[j] - p[i], 2) / p2) +
        factorL / (Math.pow(t[j] - p[i], 2) + p2);
    }
  }
  return result;
}
