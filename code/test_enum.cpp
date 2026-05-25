#include "hq/cerberus_api_gateway.hpp"
#include "hq/cerberus_slipstream.hpp"
#include <windows.h>

int main() {
    using namespace hq::cerberus::gateway;
    auto msg = ProtocolHelper::buildMessage(CerberusOpcode::HANDSHAKE_INIT, 0, 1, {});
    CerberusApiGateway gw;
    gw.setPermissionMode(0, PermissionMode::ACT);

    using namespace hq::cerberus::slipstream;
    auto smsg = SlipstreamMessage::create(SlipstreamMessageType::CMD_EXECUTE, {}, 0);
    return 0;
}
