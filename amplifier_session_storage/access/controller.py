"""Access control for session sharing."""

from ..protocol import SessionMetadata, SessionVisibility, UserMembership
from .permissions import AccessDecision, Permission


class AccessController:
    """Centralized access control for session sharing.

    Implements permission checks based on session visibility and user membership.
    """

    async def check_access(
        self,
        user_id: str,
        session: SessionMetadata,
        membership: UserMembership | None,
        required: Permission,
    ) -> AccessDecision:
        """Check if user has required permission on session.

        Args:
            user_id: User requesting access
            session: Session to check access for
            membership: User's organization/team membership (may be None)
            required: Permission level required

        Returns:
            AccessDecision with allowed status and reason
        """
        # Owner always has full access
        if session.user_id == user_id:
            return AccessDecision(allowed=True, reason="owner", permission_level=Permission.DELETE)

        # Private sessions - no access for non-owners
        if session.visibility == SessionVisibility.PRIVATE:
            return AccessDecision(allowed=False, reason="private_session")

        # Non-owners can only have READ access to shared sessions
        if required != Permission.READ:
            return AccessDecision(allowed=False, reason="write_access_owner_only")

        if membership is None:
            return AccessDecision(allowed=False, reason="no_membership")

        # Check based on visibility level
        if session.visibility == SessionVisibility.PUBLIC:
            return AccessDecision(
                allowed=True, reason="public_read", permission_level=Permission.READ
            )

        elif session.visibility == SessionVisibility.ORGANIZATION:
            if membership.org_id == session.org_id:
                return AccessDecision(
                    allowed=True, reason="org_member", permission_level=Permission.READ
                )

        elif session.visibility == SessionVisibility.TEAM:
            if session.team_ids:
                user_teams = set(membership.team_ids)
                session_teams = set(session.team_ids)
                if user_teams & session_teams:
                    return AccessDecision(
                        allowed=True,
                        reason="team_member",
                        permission_level=Permission.READ,
                    )

        return AccessDecision(allowed=False, reason="insufficient_permission")
