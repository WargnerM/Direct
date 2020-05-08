const Direct = require('../index');
const parseData = require('../util/parseData');
const getErrorFunction = require('../util/getErrorFunction');
const sumOfGaussianLorentzians = require('../functions/sumOfGaussianLorentzians')

function optimizeGLSum(xy, group, options = {}) {
  var {
    percentage = 0,
    iterations = 20,
    noiseLevel = 0,
    initialState = {}
  } = options;

  var xy2 = parseData(xy, percentage);
  if (xy2 === null || xy2[0].rows < 3) {
    return null;
  }

  var t = xy2[0];
  var yData = xy2[1];
  var maxY = xy2[2];
  var consts = [noiseLevel / maxY];
  var nL = group.length;
  var pMin = new Array(nL * 4, 1);
  var pMax = new Array(nL * 4, 1);
  var dt = Math.abs(t[0] - t[1]);
  
  for (let i = 0; i < nL; i++) {
    pMin[i] = group[i].x - dt;
    pMin[i + nL] = 0;
    pMin[i + 2 * nL] = group[i].width / 4;
    pMin[i + 3 * nL] = 0;

    pMax[i] = group[i].x + dt;
    pMax[i + nL] = 1.5;
    pMax[i + 2 * nL] = group[i].width * 4;
    pMax[i + 3 * nL] = 1;
  }
  var targetFunc = getErrorFunction(sumOfGaussianLorentzians, { x: t, y: yData }, consts);

  var result = Direct(
    targetFunc,
    pMin,
    pMax,
    { iterations },
    initialState
  );

  for (let k = 0; k < result.optimum.length; k++) {
    let pFit = result.optimum[k];
    for (let i = 0; i < nL; i++) {
      pFit[i + nL] *= maxY;
    }
    result.optimum[k] = pFit;
  }
  return result;
}

module.exports = optimizeGLSum;