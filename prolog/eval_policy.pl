% Deterministic base-vs-fine-tune promotion policy for Pi task evaluations.
% scripts/pi_eval.py generates eval_result/7 and gate/2 facts, then consults this file.

:- use_module(library(http/json)).
:- dynamic eval_result/7.
:- dynamic gate/2.

run_key(Model, Task-Repeat) :-
    eval_result(Model, Task, Repeat, _Outcome, _LatencyMs, _InputTokens, _OutputTokens).

matched_eval_sets :-
    findall(Key, run_key(baseline, Key), Baseline0),
    findall(Key, run_key(candidate, Key), Candidate0),
    sort(Baseline0, Baseline),
    sort(Candidate0, Candidate),
    Baseline \= [],
    Baseline == Candidate.

run_count(Model, Count) :-
    findall(1, eval_result(Model, _Task, _Repeat, _Outcome, _Latency, _In, _Out), Runs),
    length(Runs, Count).

pass_count(Model, Count) :-
    findall(1, eval_result(Model, _Task, _Repeat, pass, _Latency, _In, _Out), Runs),
    length(Runs, Count).

pass_rate(Model, Rate) :-
    run_count(Model, Total),
    Total > 0,
    pass_count(Model, Passed),
    Rate is Passed / Total.

task_pass_rate(Model, Task, Rate) :-
    findall(Outcome,
            eval_result(Model, Task, _Repeat, Outcome, _Latency, _In, _Out),
            Outcomes),
    Outcomes \= [],
    include(=(pass), Outcomes, Passed),
    length(Outcomes, Total),
    length(Passed, PassCount),
    Rate is PassCount / Total.

regression(Task) :-
    task_pass_rate(baseline, Task, Baseline),
    task_pass_rate(candidate, Task, Candidate),
    Candidate < Baseline.

improvement(Task) :-
    task_pass_rate(baseline, Task, Baseline),
    task_pass_rate(candidate, Task, Candidate),
    Candidate > Baseline.

regression_count(Count) :-
    findall(Task, regression(Task), Tasks0),
    sort(Tasks0, Tasks),
    length(Tasks, Count).

mean_latency(Model, Mean) :-
    findall(Latency,
            eval_result(Model, _Task, _Repeat, _Outcome, Latency, _In, _Out),
            Values),
    Values \= [],
    sum_list(Values, Total),
    length(Values, Count),
    Mean is Total / Count.

latency_ratio(Ratio) :-
    mean_latency(baseline, Baseline),
    mean_latency(candidate, Candidate),
    ( Baseline =:= 0.0 ->
        ( Candidate =:= 0.0 -> Ratio = 1.0 ; Ratio = 1.0e100 )
    ; Ratio is Candidate / Baseline
    ).

gate_value(Name, Default, Value) :-
    ( gate(Name, Configured) -> Value = Configured ; Value = Default ).

promotable :-
    matched_eval_sets,
    pass_rate(baseline, BaselineRate),
    pass_rate(candidate, CandidateRate),
    gate_value(min_pass_rate_delta, 0.0, MinDelta),
    CandidateRate >= BaselineRate + MinDelta,
    regression_count(Regressions),
    gate_value(max_regressions, 0, MaxRegressions),
    Regressions =< MaxRegressions,
    latency_ratio(LatencyRatio),
    gate_value(max_latency_ratio, 1.5, MaxLatencyRatio),
    LatencyRatio =< MaxLatencyRatio.

promotion_verdict(Dict) :-
    pass_rate(baseline, BaselineRate),
    pass_rate(candidate, CandidateRate),
    regression_count(RegressionCount),
    findall(Task, regression(Task), Regressions0),
    sort(Regressions0, Regressions),
    findall(Task, improvement(Task), Improvements0),
    sort(Improvements0, Improvements),
    latency_ratio(LatencyRatio),
    ( promotable -> Verdict = promote ; Verdict = reject ),
    ( matched_eval_sets -> Matched = true ; Matched = false ),
    Dict = _{
        verdict: Verdict,
        matched_eval_sets: Matched,
        baseline_pass_rate: BaselineRate,
        candidate_pass_rate: CandidateRate,
        pass_rate_delta: CandidateRate-BaselineRate,
        regression_count: RegressionCount,
        regressions: Regressions,
        improvements: Improvements,
        mean_latency_ratio: LatencyRatio
    }.

promotion_json :-
    ( promotion_verdict(Dict) ->
        json_write_dict(current_output, Dict, [width(0)]), nl
    ; json_write_dict(current_output,
                      _{verdict: reject, matched_eval_sets: false, reason: no_complete_eval_set},
                      [width(0)]), nl
    ).
