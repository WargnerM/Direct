
function sumOfPseudoVoigt(t, p, c) {
    var nL = p.length / 4;
    var factorG;
    var factorL;
    var cols = t.length;
    var p2;
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

module.exports = sumOfPseudoVoigt
