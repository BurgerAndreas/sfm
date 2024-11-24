
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, TargetConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from torchcfm.optimal_transport import OTPlanSampler

def get_flowmatching(args, sigma=None, **kwargs):
    # passed sigma > args > default
    if sigma is None:
        sigma = args.get("sigma", 0.01)
    # get flowmatching
    if args.classcond:
        assert args.use_ot, "Class conditional flow matching requires use_ot=True"
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif args.fmloss == "lipman":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif args.use_ot:
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif args.fmloss == "cfm":
        return ConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Unknown flow matching loss: {args.fmloss}")