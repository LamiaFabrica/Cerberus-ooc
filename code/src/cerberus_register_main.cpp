/// @file cerberus_register_main.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Cerberus Registration CLI — First-Run Local Database Provisioning
/// ===================================================================
///
/// Walks the user through:
///   1. Passphrase entry (hidden input)
///   2. Memorable word entry (8–40 chars)
///   3. Generates system PIN + hardware anchor
///   4. Derives AES-256-GCM key via Argon2id (MANDATORY — HMAC fallback removed)
///   5. Creates encrypted Local Maintenance DB
///   6. Displays confirmation + recovery warnings
///
/// Usage:
///   cerberus_register [db_path]
///   (default db_path: ./cerberus_local.db)
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include "hq/cerberus_first_run.hpp"
#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_psiforcedb_security.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <conio.h>
#else
#  include <termios.h>
#  include <unistd.h>
#endif

namespace {

// Hidden passphrase input (cross-platform)
std::string read_hidden_line() {
    std::string line;
#ifdef _WIN32
    char ch;
    while ((ch = static_cast<char>(_getch())) != '\r' && ch != '\n') {
        if (ch == '\b' && !line.empty()) {
            line.pop_back();
            std::fputc('\b', stdout);
            std::fputc(' ', stdout);
            std::fputc('\b', stdout);
        } else if (ch != '\b') {
            line.push_back(ch);
            std::fputc('*', stdout);
        }
    }
    std::fputc('\n', stdout);
#else
    termios oldt{}, newt{};
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    std::getline(std::cin, line);
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
#endif
    return line;
}

void clear_screen() {
    // Cross-platform screen clear — avoids std::system() warnings on WSL
#ifdef _WIN32
    std::cout << std::string(50, '\n');
#else
    // POSIX terminals support ANSI escape; Windows cmd does not
    std::cout << "\033[2J\033[H";
#endif
}

void print_banner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CERBERUS — Local Database Registration                          ║\n";
    std::cout << "║  First-Run Provisioning for Encrypted Local Maintenance DB       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_warning() {
    std::cout << "\n";
    std::cout << "⚠️  CRITICAL RECOVERY WARNING\n";
    std::cout << "─────────────────────────────────────────────────────────────────\n";
    std::cout << "Your Local Maintenance Database uses end-to-end encryption.\n\n";
    std::cout << "  • NO ONE can recover your data if you lose your passphrase.\n";
    std::cout << "  • Not LamiaFabrica. Not PsiForceDB. Not even the developer.\n";
    std::cout << "  • There is NO \"forgot password\" button. There is NO backdoor.\n";
    std::cout << "  • If you forget your passphrase, your data is PERMANENTLY LOST.\n\n";
    std::cout << "The only way to continue is to write down your passphrase\n";
    std::cout << "and memorable word in a safe, offline location.\n\n";
}

} // namespace

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    clear_screen();
    print_banner();

    // ── Database path ──
    std::filesystem::path db_path = std::filesystem::current_path() / "cerberus_local.db";
    if (argc > 1) db_path = argv[1];

    // ── Check if already registered ──
    hq::cerberus::privacy::FirstRun fr;
    if (fr.is_already_registered(db_path)) {
        std::cout << "❌ A database already exists at: " << db_path.string() << "\n";
        std::cout << "   If you want to re-register, delete it first.\n";
        std::cout << "   WARNING: Deleting the database destroys ALL local data.\n\n";
        return 1;
    }

    // ── Warn about recovery ──
    print_warning();
    std::cout << "Type 'yes' to acknowledge and continue: ";
    std::string ack;
    std::getline(std::cin, ack);
    if (ack != "yes") {
        std::cout << "\nRegistration cancelled. No changes made.\n";
        return 0;
    }

    clear_screen();
    print_banner();

    // ── Step 1: Passphrase ──
    std::string passphrase;
    while (true) {
        std::cout << "Step 1/3 — Enter a strong passphrase (minimum 12 characters): ";
        passphrase = read_hidden_line();
        if (passphrase.size() >= 12) break;
        std::cout << "   Too short. Minimum 12 characters required.\n";
    }

    // ── Step 2: Memorable word ──
    std::string memorable_word;
    while (true) {
        std::cout << "Step 2/3 — Enter a memorable word (8–40 chars, e.g. 'RedFox2026'): ";
        std::getline(std::cin, memorable_word);
        // Note: MemorableWord is in hq::cerberus::security namespace
        auto err = hq::cerberus::privacy::MemorableWord::validate(memorable_word);
        if (err.empty()) break;
        std::cout << "   Invalid: " << err << "\n";
    }

    clear_screen();
    print_banner();
    std::cout << "Step 3/3 — Provisioning encrypted database...\n\n";

    // ── Register ──
    auto result = fr.register_new_install(
        passphrase, memorable_word, db_path, /*psi_reachable=*/false);

    if (result.db_key.empty()) {
        std::cout << "❌ Registration FAILED: " << result.diagnostic << "\n";
        std::cout << "\nPossible causes:\n";
        std::cout << "  • LFSSL library not found (cerberus_lfssl.dll / libcerberus_lfssl.so)\n";
        std::cout << "  • LFSSL found but Argon2id export is missing (incomplete build)\n";
        std::cout << "  • Memory allocation failure (Argon2id m_cost=65536 requires ~64MB)\n";
        std::cout << "  • Disk write permissions denied\n";
        return 1;
    }

    // ── Success display ──
    std::cout << "✅ Registration SUCCESSFUL\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n\n";

    std::cout << "  Node ID:         " << result.node_id << "\n";
    std::cout << "  PIN (system):    " << result.issued_pin << "\n";
    std::cout << "  Database path:   " << db_path.string() << "\n";
    std::cout << "  Encryption:      AES-256-GCM (LFSSL)\n";
    std::cout << "  Key derivation:  Argon2id (memory-hard, OWASP 2023 recommended)\n";
    std::cout << "  Trust policy:    server_isolated, no recovery, rebuild only\n";
    std::cout << "  RBPC:            3 failed PIN attempts = permanent burn\n\n";

    std::cout << "⚠️  WRITE THIS DOWN IN A SAFE PLACE:\n";
    std::cout << "   Passphrase:      [hidden — you just typed it]\n";
    std::cout << "   Memorable word:  " << memorable_word << "\n";
    std::cout << "   System PIN:      " << result.issued_pin << "\n\n";

    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "Your data is now encrypted. Even the developer cannot read it.\n";
    std::cout << "If you forget any of the above, your data is PERMANENTLY LOST.\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n\n";

    return 0;
}
