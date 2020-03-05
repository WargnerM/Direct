
/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Matrix} fun - dataset or covariance matrix.
 * @param {Array} xU - dataset or covariance matrix.
 * @param {Array} xL - dataset or covariance matrix.
 * @param {Object} [options]
 * @param {number} [options.printLevel] - Parameter to show information.
 * @param {Object} [options.global] - Parameter with information.
 * */


export function gblSolve(fun, xL, xU, options = {}, entries = {}) {
  const opts = {
    iterations: 50,
    epsilon: 1e-4,
    tol: 0.01,
  };

  let global = entries;
  options = Object.assign({}, options, opts);

  if (fun === undefined || xL === undefined || xU === undefined) {
    throw new Error('There is something undefined');
  }

  if (xL.length !== xU.length) {
    throw new Error('Lower bounds and Upper bounds for x are not of the same length');
  }

  // *** Step 1: Inicialization
  let funCalls = 0;
  let convFlag = 0;
  let n = xL.length;
  let tolle = 1e-16;
  let tolle2 = 1e-12;
  let dMin = global.dMin

  let F, m, D, L, d, fMin, epsilon, E, minIndex, iMin, f0;
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
    let test = F.map((x) => x - (fMin + E));
    let difference = [];

    for (let i = 0; i < test.length; i++) {
      difference[i] = test[i] - D[i];
    }
    minIndex = Math.min(...difference);
    let C = [];
    for (let i = 0; i < m; i++) {
      C[i] = global.C[i].map((x) => x - (xL[i] / (xU[i] - xL[i])));
    }
  } else {
    m = 1;
    let C = new Array(n).fill(0.5);
    let xM = [];
    for (let i = 0; i < xL.length; i++) {
      xM[i] = xL[i] + C[i] * (xU[i] - xL[i]);
    }
    let fMin = fun(xM);
    f0 = fMin;
    funCalls = funCalls + 1;
    iMin = 1;
    let L = new Array(n).fill(0.5);
    let D = Math.sqrt(
      L.reduce((a, b) => a + Math.pow(b, 2))
    );
    F = [fMin];
    d = D;
    dMin = fMin;
  }
}

