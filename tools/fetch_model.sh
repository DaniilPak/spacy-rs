#!/usr/bin/env sh
# Download and unpack an official spaCy pipeline (MIT-licensed weights from
# github.com/explosion/spacy-models) for spacy-rs.
#   tools/fetch_model.sh [name] [version] [dest]
set -eu
NAME="${1:-en_core_web_sm}"
VERSION="${2:-3.8.0}"
DEST="${3:-tests/models}"
URL="https://github.com/explosion/spacy-models/releases/download/${NAME}-${VERSION}/${NAME}-${VERSION}-py3-none-any.whl"
TMP="$(mktemp -d)"
echo "Downloading ${URL}"
curl -fsSL -o "${TMP}/model.whl" "${URL}"
mkdir -p "${TMP}/unzipped" "${DEST}"
unzip -q "${TMP}/model.whl" -d "${TMP}/unzipped"
rm -rf "${DEST:?}/${NAME}"
mv "${TMP}/unzipped/${NAME}/${NAME}-${VERSION}" "${DEST}/${NAME}"
rm -rf "${TMP}"
echo "Pipeline unpacked to ${DEST}/${NAME}"
