
(
{
  var sig, env, filter, delay, mix, verb;

  sig = WhiteNoise.ar(mul:1.0, add:0.0);
  env = SinOsc.kr(0.15).range(0.3, 1.0);

  filter = LPF.ar(in:sig * env, freq: SinOsc.kr(0.1).range(250, LFNoise1.kr(0.01).range(400, 500)), mul:1.0, add:0.0)!2;
  delay = CombN.kr(in:0.0, maxdelaytime:5.2, delaytime:3.2, decaytime:10.0, mul:1.0, add:0.0);

  mix = (0.5 * delay) + filter;
  
  verb = FreeVerb.ar(in:mix, mix:0.53, room:0.5, damp:0.7, mul:1.0, add:0.0);

  Limiter.ar(verb, 0.9)
}.play
)

s.boot
