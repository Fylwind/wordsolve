function assertEq(context, result, expected) {
    if (result != expected) {
        console.error(context, result, expected);
    }
}

for (const [query, candidate, response] of [
    ["abcde", "abcde", "22222"],
    ["apcdq", "abcde", "20220"],
    ["pbqde", "abcde", "02022"],
    ["bbbde", "abcbd", "12010"],
    ["bbcdc", "adcbb", "11210"],
    ["pqrst", "abcbd", "00000"],
]){
    assertEq(unpackWord(packWord(query)) == query);
    assertEq(unpackWord(packWord(candidate)) == candidate);
    assertEq(unpackResponse(packResponse(response)) == response);
    assertEq(
        {query, candidate},
        unpackResponse(computeResponse(packWord(query), packWord(candidate))),
        response);
}
