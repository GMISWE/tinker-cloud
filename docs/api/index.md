---
layout: default
title: API Reference
nav_order: 6
has_children: true
---

# API Reference

Complete HTTP API documentation for tinkercloud.

## Overview

Tinkercloud exposes a RESTful API on port 8000 that is compatible with the Tinker API specification.

### Authentication

All endpoints require API key authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: slime-dev-key" http://localhost:8000/api/v1/health
```

### Sections

- [API Endpoints](endpoints.md) - Complete HTTP endpoint reference
- [Types](types.md) - Request and response schemas
