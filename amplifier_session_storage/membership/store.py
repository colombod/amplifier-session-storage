"""Membership data storage."""

from typing import Any, Protocol

from ..protocol import Organization, Team, UserMembership


class CosmosClientProtocol(Protocol):
    """Protocol for Cosmos DB client operations."""

    async def read_item(
        self,
        container_name: str,
        item_id: str,
        partition_key: str,
    ) -> dict[str, Any] | None: ...

    async def query_items(
        self,
        container_name: str,
        query: str,
        parameters: list[dict[str, Any]] | None = None,
        partition_key: str | None = None,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def upsert_item(
        self,
        container_name: str,
        item: dict[str, Any],
        partition_key: str,
    ) -> dict[str, Any]: ...


# Container names
ORGANIZATIONS_CONTAINER = "organizations"
TEAMS_CONTAINER = "teams"
USER_MEMBERSHIPS_CONTAINER = "user_memberships"


class MembershipStore:
    """Store for organization/team/membership data.

    Provides access to user membership information needed for
    access control decisions.
    """

    def __init__(self, client: CosmosClientProtocol):
        """Initialize membership store.

        Args:
            client: Cosmos DB client wrapper
        """
        self.client = client

    async def get_membership(self, user_id: str) -> UserMembership | None:
        """Get user's membership.

        Args:
            user_id: User to look up

        Returns:
            User membership or None if not found
        """
        result = await self.client.read_item(USER_MEMBERSHIPS_CONTAINER, user_id, user_id)
        if result:
            return UserMembership.from_dict(result)
        return None

    async def get_organization(self, org_id: str) -> Organization | None:
        """Get organization by ID.

        Args:
            org_id: Organization ID

        Returns:
            Organization or None if not found
        """
        result = await self.client.read_item(ORGANIZATIONS_CONTAINER, org_id, org_id)
        if result:
            return Organization.from_dict(result)
        return None

    async def get_teams_for_org(self, org_id: str) -> list[Team]:
        """Get all teams in an organization.

        Args:
            org_id: Organization ID

        Returns:
            List of teams in the organization
        """
        results = await self.client.query_items(
            TEAMS_CONTAINER,
            "SELECT * FROM c WHERE c.org_id = @org_id",
            [{"name": "@org_id", "value": org_id}],
            org_id,
        )
        return [Team.from_dict(doc) for doc in results]

    async def get_teams_for_user(self, user_id: str) -> list[Team]:
        """Get all teams a user belongs to.

        Args:
            user_id: User ID

        Returns:
            List of teams the user is a member of
        """
        membership = await self.get_membership(user_id)
        if not membership or not membership.team_ids:
            return []

        teams = []
        for team_id in membership.team_ids:
            result = await self.client.read_item(TEAMS_CONTAINER, team_id, membership.org_id)
            if result:
                teams.append(Team.from_dict(result))
        return teams

    async def set_membership(self, membership: UserMembership) -> UserMembership:
        """Create or update user membership.

        Args:
            membership: Membership to create/update

        Returns:
            The saved membership
        """
        doc = membership.to_dict()
        doc["id"] = membership.user_id
        doc["_type"] = "user_membership"
        await self.client.upsert_item(USER_MEMBERSHIPS_CONTAINER, doc, membership.user_id)
        return membership
