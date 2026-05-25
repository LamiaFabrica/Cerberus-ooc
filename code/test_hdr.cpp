#include "hq/cerberus_api_gateway.hpp"
#include "hq/cerberus_slipstream.hpp"
int main() {
    using namespace hq::cerberus::gateway;
    using namespace hq::cerberus::slipstream;
    auto msg1 = ProtocolHelper::buildMessage(CerberusOpcode::SESSION_CLOSE, 0, 0, {});
    auto msg2 = SlipstreamMessage::create(SlipstreamMessageType::CMD_EXECUTE, {}, 0);
    return 0;
}
