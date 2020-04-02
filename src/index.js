'use strict';

let Matrix = require('ml-matrix').Matrix;
let conhull = require('../src/conhull.js');

/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} fun - dataset or covariance matrix.
 * @param {Array} xU - dataset or covariance matrix.
 * @param {Array} xL - dataset or covariance matrix.
 * @param {Object} [options]
 * @param {number} [options.printLevel] - Parameter to show information.
 * @param {Object} [options.global] - Parameter with information.
 * */


function Direct(fun, xL, xU, options = {}, entries = {}) {
  const opts = {
    iterations: 50,
    epsilon: 1e-4,
    tol: 0.01,
  };

  let global = entries;
  options = Object.assign({}, opts, options);

  if (fun === undefined || xL === undefined || xU === undefined) {
    throw new Error('There is something undefined');
  }

  if (xL.length !== xU.length) {
    throw new Error('Lower bounds and Upper bounds for x are not of the same length');
  }
  // if (global.C) {
  //   global.C = Matrix.checkMatrix(global.C);
  // }
  
  //-------------------------------------------------------------------------
  //                        STEP 1. Initialization
  //-------------------------------------------------------------------------
  let iterations = options.iterations;
  let epsilon = options.epsilon;
  let tol = options.tol;
  let funCalls = 0;
  let n = xL.length;
  let tolle = 1e-16;
  let tolle2 = 1e-12;
  let dMin = global.dMin;
  let diffBorders = xU.map((x, i) => x - xL[i]);
 
  let F, m, D, L, d, fMin, E, iMin, minIndex, f0, C;
  if (global.C && global.C.length !== 0) {
    F = global.F;
    m = F.length;
    D = global.D;
    L = global.L;
    d = global.d;
    dMin = global.dMin;
    epsilon = options.epsilon;
    fMin = Math.min(...F);
    E = options.epsilon * Math.abs(fMin) > 1e-8 ? options.epsilon * Math.abs(fMin) : 1e-8;
    let item = [];
    for (let i = 0; i < F.length; i++) {
      item[i] = (F[i] - (fMin + E))/D[i]
    }
    iMin = item.findIndex(x => x == Math.min(...item));
    for (let i = 0; i < m; i++) {
      C[i] = global.C(i).map((x) => x - (xL / (xU - xL)));
    }
  } else {
    m = 0;
    C = [new Array(n).fill(0.5)];
    let xM = [];
    for (let i = 0; i < xL.length; i++) {
      xM[i] = xL[i] + C[0][i] * diffBorders[i];
    }
    fMin = fun(xM);
    funCalls = funCalls + 1;
    iMin = 0;
    L = [ new Array(n).fill(0.5) ];
    D = [ Math.sqrt(n * Math.pow(0.5, 2)) ];
    F = [ fMin ];
    d = D;
    dMin = [fMin];
  }

  let t = 1;

  //-------------------------------------------------------------------------
  //                          Iteration loop
  //-------------------------------------------------------------------------
  
  while (t < options.iterations) {
    //----------------------------------------------------------------------
    //  STEP 2. Identify the set S of all potentially optimal rectangles
    //----------------------------------------------------------------------
    let S = [];
    let S1 = [];
    let S2 = [];
    let S3 = [];
    let idx = d.findIndex((e) => e === D[iMin]);
    let a = 0;
    for (let i = idx; i < d.length; i++) {
      let idx2 = [];
      for (let f = 0; f < F.length; f++) {
        if (F[f] === dMin[i]) {
          if (D[f] === d[i]) idx2.push(f);
        }
      }
      S1 = S1.concat(idx2);
    }
    // console.log('s1',S1)
    if (d.length - idx > 1) { // toTest the condition
      let a1 = D[iMin];
      let b1 = F[iMin];
      let a2 = d[d.length - 1];
      let b2 = dMin[d.length - 1];
      let slope = (b2 - b1) / (a2 - a1);
      let constant = b1 - slope * a1;
      for (let i = 0; i < S1.length; i++) {
        let j = S1[i];
        if (F[j] <= slope * D[j] + constant + tolle2) {
          S2.push(j);
        }
      }
      let xx = new Array(S2.length);
      let yy = new Array(S2.length);
      for (let i = 0; i < S2.length; i++) {
        xx[i] = [ D[S2[i]] ];
        yy[i] = [ F[S2[i]] ];
      }
      // console.log('conhull input',xx, yy)
      let h = conhull(xx, yy);
      // console.log('conhull output', h)
      S3 = new Array(h.length);
      for (let i = 0; i < h.length; i++) {
        S3[i] = S2[h[i]];
      }
    } else {
      S3 = S1;
    }
    S = S3;
    console.log('S', S)
    //--------------------------------------------------------------
    // STEPS 3,5: Select any rectangle j in S
    //--------------------------------------------------------------
    for (let por = 0; por < S.length; por++) {
      let j = S[por];
      let maxL = Math.max(...L[j]);
      console.log('maxL', maxL);
      let I = [];
      for (let i = 0; i < L[j].length; i++) {
        if (Math.abs(L[j][i] - maxL) < tolle) I.push(i);
      }
      console.log('I vector', I)
      let delta = (2 * maxL) / 3;
      let w = [];
      for (let r = 0; r < I.length; r++) {
        let i = I[r]; //I[r] + 1; why plus one? i is the index of the dimension that will be splitted
        let cm1 = C[j].slice();
        let cm2 = C[j].slice();
        cm1[i] += delta;
        cm2[i] -= delta;

        let xm1 = []; let xm2 = [];
        for (let i = 0; i < cm1.length; i++) {
          xm1[i] = xL[i] + (cm1[i] * diffBorders[i]);
          xm2[i] = xL[i] + (cm2[i] * diffBorders[i]);
        }
        let fm1 = fun(xm1);
        let fm2 = fun(xm2);
        funCalls += 2;
        w[r] = [Math.min(fm1, fm2), r];
        C.push(cm1, cm2);
        F.push(fm1, fm2);
      }
        // console.log('C', C)
      let b = w.sort((a, b) => a[0] - b[0]);

      for (let r = 0; r < I.length; r++) {
        let u = I[b[r][1]];
        let ix1 = m + (2 * (b[r][1]) + 1) - 1;
        let ix2 = m + (2 * (b[r][1]) + 1);
        L[j][u] = delta / 2;
        L[ix1] = L[j].slice();
        L[ix2] = L[j].slice();
        let sumSquare = 0
        for (let i = 0; i < L[j].length; i++) {
          sumSquare += Math.pow(L[j][i], 2);
        }
        D[j] = Math.sqrt(sumSquare);
        D[ix1] = D[j];
        D[ix2] = D[j];
      }
      m += 2 * I.length;
    }
    //--------------------------------------------------------------
    //                  Update
    //--------------------------------------------------------------
    fMin = Math.min(...F);
    E = options.epsilon * Math.abs(fMin) > 1e-8 ? options.epsilon * Math.abs(fMin) : 1e-8;
    
    let prevValue = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < F.length; i++) {
      let newValue = (F[i] - (fMin + E))/D[i];
      if (newValue < prevValue) {
        iMin = i;
        prevValue = newValue;
      }
    }

    d = D.slice();
    for (let i = 0; i !== d.length; i++) {
      let dTmp = d[i];
      let idx = [];
      for (let di = 0; di < d.length; di++) {
        if (d[di] !== dTmp) idx.push(di);
      }
      let newD = new Array(idx.length + 1);
      newD[0] = dTmp;
      for (let k = 0, kk = 1; k < idx.length; k++) {
        newD[kk++] = d[idx[k]];
      }
      d = newD;
    }
    
    d = d.sort((a, b) => a - b);

    dMin = new Array(d.length);
    for (let i = 0; i < d.length; i++) {
      let minIndex;
      let minValue = Number.MAX_SAFE_INTEGER;
      for (let k = 0; k < D.length; k++) {
        if (D[k] === d[i]) {
          if (F[k] < minValue) {
            minValue = F[k];
            minIndex = k;
          }
        }
      }
      // console.log('minIndex',minIndex)
      dMin[i] = F[minIndex];
    }
    console.log('F', F )
    console.log('C', C);
    console.log('dMin', dMin)
    console.log('D',D )
    console.log('d', d)
    console.log('L', L)
    console.log('fMin', fMin)
    t++
    // console.log(dMin)
  }
  // console.log(dMin)
  //--------------------------------------------------------------
  //                  Saving results
  //--------------------------------------------------------------

  let result = {};
  result.fK = fMin; // Best function value
  result.iterations = t; // Number of iterations
  
  
  for (let j = 0; j < m; j++) { // Transform to original coordinates
    for (let i = 0; i < xL.length; i++) {
      C[j][i] = xL[i] + C[j][i] * diffBorders[i];
    }
  }
  
  // let global = {};
  global.C = C; // All sampled points in original coordinates
  global.F = F; // All function values computed
  global.D = D; // All distances
  global.L = L; // All lengths
  global.d = d;
  global.dMin = dMin;
  global.funCalls = funCalls;
  result.global = global;
  // Find all points i with F(i)=f_min
  let index = 0;
  let xK = [];
  for (let i = 0; i < F.length; i++) {
    if (F[i] === fMin) xK.push(C[i]);
  }
  result.xK = xK;
  return result;
}

// //--------------------------------------------------------
// //   Testing the algorithm with benchmark functions
// //-----------------------------------------------------

function testFunction(x) {
  let a = (x[1] - 5 * Math.pow(x[0], 2) / (4 * Math.pow(Math.PI, 2)) + 5 * x[0] / Math.PI - 6);
  let b = (10 * (1 - 1 / (8 * Math.PI)) * Math.cos(x[0]) + 10);
  let result = Math.pow(a, 2) + b;
  return result;
}

let xL = [-5, 0];
let xU = [10, 15];
let GLOBAL = { iterations: 20};
let result = Direct(testFunction, xL, xU, GLOBAL);
console.log('-___----------_____RESULT-____----------___');
console.log(result);